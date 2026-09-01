import math
import random
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
import joblib

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


MAX_FRAGMENTS = 60

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class MACCSModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()

        self.max_fragments = MAX_FRAGMENTS
        self.projection_layer = nn.Sequential(
            nn.Linear(177, d_model),
            Swish(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 167),
        )

    def padding(self, B, batch_submaccs, device):
        tokens = torch.zeros(B, self.max_fragments, 177, device=device)
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
            charge = spec.get("charge")
            bit_map = (
                fp_bit_map_p_mode if charge == 1 or "positiv" in ionmode
                else fp_bit_map_n_mode if charge == -1 or "negativ" in ionmode
                else None
            )

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
        ionmode = spec.get("ionmode", "").lower() 
        charge = spec.get("charge", "")
        if "positiv" in ionmode or "negativ" in ionmode:
            return spec
        if charge == 1 or charge == -1:
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
    """
    Inference wrapper that returns both MACCS and formula predictions.
    """

    def __init__(
        self,
        model_path: str,
        fp_bit_map_p_mode: str,
        fp_bit_map_n_mode: str,
        device: str = "cpu",
    ):
        self.fp_bit_map_p_mode = fp_bit_map_p_mode
        self.fp_bit_map_n_mode = fp_bit_map_n_mode
        self.device = device

        d_model    = 512
        nhead      = 8
        num_layers = 5
        dropout    = 0.15

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = MACCSModel(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        ).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # ------------------------------------------------------------------
    def calc_fp(self, mgf):
        """
        Returns
        -------
        maccs_preds   : torch.Tensor of shape (N, 167)  — binary MACCS bits
        formula_preds : torch.Tensor of shape (N, FORMULA_DIM) — normalised element counts
        """
        dataset = MS2Data(
            mgf,
            self.fp_bit_map_p_mode,
            self.fp_bit_map_n_mode,
            mode="eval",
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        maccs_preds = []
        with torch.no_grad():
            for submaccs, _ in tqdm(loader, desc="MACCS Prediction"):
                maccs_logits = self.model(submaccs)
                maccs_binary = (torch.sigmoid(maccs_logits) >= 0.5).float()
                maccs_preds.append(maccs_binary)

        return torch.cat(maccs_preds, dim=0)
    

    def calc_tox(self, mgf, tox_model_path):

        tox_model_dir = os.listdir(tox_model_path)
        true_fps = pd.read_csv(tox_model_path+"/../"+"toxcast_maccs_fps.csv")
        train_fp_cache = {}

        maccs = self.calc_fp(mgf).to("cpu")

        predictions = []
        probabilities = []
        balanced_accuracies = []
        aeids = []
        similarities = []

        for model_id in tqdm(tox_model_dir, desc="Tox Prediction"):
            if "FS_RandomForest" not in os.listdir(tox_model_path+"/"+model_id):
                continue
            filter_ = joblib.load(tox_model_path+"/"+model_id+"/"+"FS_RandomForest"+"/preprocessing_model.joblib")
            classifier = joblib.load(tox_model_path+"/"+model_id+"/"+"FS_RandomForest/XGBClassifier/best_estimator_train.joblib")
            metrics = pd.read_csv(tox_model_path+"/"+model_id+"/"+"FS_RandomForest/XGBClassifier/metrics.csv").iloc[0]
            balanced_accuracies.append(metrics.get("balanced_accuracy"))
            selected_chemicals = pd.read_csv(tox_model_path+"/"+model_id+"/"+"selected_chemicals.csv")
            #maccs = maccs.squeeze(0) 
            maccs_df = pd.DataFrame(maccs, columns=filter_.feature_names_in_)

            filtered_maccs = filter_.transform(maccs_df)


            if hasattr(filter_, "get_feature_names_out"):
                final_features = filter_.get_feature_names_out()
            else:
                final_features = [f"f{i}" for i in range(filtered_maccs.shape[1])]
            if model_id not in train_fp_cache:
                selected_ids = selected_chemicals["DTXSID"]
                filtered_true = true_fps[true_fps["DTXSID"].isin(selected_ids)]
                filtered_true = filtered_true.loc[:, final_features]
                train_fp_cache[model_id] = filtered_true.to_numpy(dtype=bool)

            #sim_matrix = cosine_similarity(filtered_maccs, filtered_true)
            #similarity = sim_matrix.max(axis=1)
            X_new_bool = filtered_maccs.astype(bool, copy=False)
            X_train_bool = filtered_true.to_numpy(dtype=bool)
            max_sim = np.zeros(X_new_bool.shape[0])
            chunk_size = 1000
            for i in range(0, X_train_bool.shape[0], chunk_size):
                chunk = X_train_bool[i:i+chunk_size]
                jaccard_sim = 1 - pairwise_distances(
                    X_new_bool,
                    chunk,
                    metric="jaccard",
                )

                max_sim = np.maximum(max_sim, jaccard_sim.max(axis=1))
            similarities.append(max_sim)

            pred = classifier.predict(filtered_maccs)
            probs = classifier.predict_proba(filtered_maccs)[:, 1]

            probabilities.append(probs)
            predictions.append(pred)
            aeids.append(model_id)

        pred_df = pd.DataFrame({
            #"chem_id": None,
            "aeid": aeids, 
            "prediction": predictions,
            "probability": probabilities,
            "similarity": similarities,
            "model_BA": balanced_accuracies,

        })

        return pred_df