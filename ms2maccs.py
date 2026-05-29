import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import rdkit.Chem as Chem
from rdkit.Chem import MACCSkeys

from matchms.importing import load_from_mgf

import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle


MAX_FRAGMENTS = 60

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MACCSModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()

        self.max_fragments = MAX_FRAGMENTS
        self.projection_layer = nn.Sequential(
            nn.Linear(167, d_model),
            nn.LayerNorm(d_model),
            Swish(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 167),
        )

    def padding(self, B, batch_submaccs, device):
        tokens = torch.zeros(B, self.max_fragments, 167, device=device)
        pad_mask = torch.ones(B, self.max_fragments + 1, dtype=torch.bool, device=device)
        pad_mask[:, 0] = False

        for b, submaccs in enumerate(batch_submaccs):
            n = min(len(submaccs), self.max_fragments)
            if n > 0:
                tokens[b, :n] = torch.stack(submaccs[:n]).to(device)
                pad_mask[b, 1:n + 1] = False

        return tokens, pad_mask
        

    def forward(self, batch_submaccs):
        device = self.cls_token.device
        B = len(batch_submaccs)
        tokens, pad_mask = self.padding(B, batch_submaccs, device)
        projection = self.projection_layer(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        cls_emb = self.transformer(
            torch.cat([cls, projection], dim=1),
            src_key_padding_mask=pad_mask,
        )
        maccs = self.head(cls_emb[:, 0, :])

        return maccs


class MS2Data(Dataset):
    def __init__(self, mgf, fp_bit_map_p_mode_path, fp_bit_map_n_mode_path, mode="train"):
        path = Path(mgf)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path.name}")
        
        with open(fp_bit_map_p_mode_path, "rb") as f:
            fp_bit_map_p_mode = pickle.load(f)
        with open(fp_bit_map_n_mode_path, "rb") as f:
            fp_bit_map_n_mode = pickle.load(f)

        self.data = []
        for spec in tqdm(load_from_mgf(str(path)), f"Processing {path.name}"):

            spec = self.preprocess(spec)
            if not spec:
                print("No ionmode given")
                continue

            if mode == "train":
                smiles = spec.get("smiles")
                mol = Chem.MolFromSmiles(smiles)
                maccs = torch.tensor(np.array(MACCSkeys.GenMACCSKeys(mol)), dtype=torch.float32)

            else:
                maccs = torch.tensor([0]*167, dtype=torch.float32)

            mzs = spec.peaks.mz.round(2)
            intensities = spec.peaks.intensities
            sorted_peaks = sorted(zip(mzs, intensities), key=lambda x: x[1], reverse=True)

            submaccs = []
            ionmode = spec.get("ionmode")
            bit_map = fp_bit_map_p_mode if ionmode == "positive" else (
                      fp_bit_map_n_mode if ionmode == "negative" else None)

            for mz, _ in sorted_peaks:
                if mz not in bit_map:
                    continue
                vec = torch.tensor(bit_map[mz], dtype=torch.float32)
                vec = vec / vec.max().clamp(min=1e-8)
                submaccs.append(vec)
                if len(submaccs) == MAX_FRAGMENTS:
                    break

            self.data.append((submaccs, maccs))

    def preprocess(self, spec):
        ionmode = spec.get("ionmode")
        if ionmode == "positive" or ionmode == "negative":
            return spec
        return None 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    submaccs = [item[0] for item in batch]
    maccs = torch.stack([item[1] for item in batch])
    return submaccs, maccs


class MS2MACCS:
    def __init__(self, model_path, fp_bit_map_p_mode, fp_bit_map_n_mode, device="cpu"):
        self.fp_bit_map_p_mode = fp_bit_map_p_mode
        self.fp_bit_map_n_mode = fp_bit_map_n_mode
        self.device = device

        d_model = 512
        nhead = 8
        num_layers = 5
        dropout = 0.15
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = MACCSModel(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, mgf):
        ms2dataset = MS2Data(mgf, self.fp_bit_map_p_mode, self.fp_bit_map_n_mode, mode="eval")
        ms2loader = DataLoader(ms2dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        maccs_preds = []
        with torch.no_grad():
            for submaccs, _ in tqdm(ms2loader, desc="Prediction"):
                logits = self.model(submaccs)
                #bit_proba = torch.sigmoid(logits)
                bit_binary  = (torch.sigmoid(logits) >= 0.5).float()
                maccs_preds.append(bit_binary)

        return torch.cat(maccs_preds, dim=0)

