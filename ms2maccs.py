import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from matchms.importing import load_from_mgf
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FRAGMENTS    = 64
MACCS_BITS       = 167
FORMULA_DIM      = 10
USE_FORMULA_ATTN = False
SMILES_KEY       = "smiles"
FORMULA_KEY      = "formula"


################################################################
# ─────────────────────  Preprocessing  ──────────────────────#
################################################################

def preprocess(spec):
    """Minimal guard — extend with your own normalisation if needed."""
    if spec is None:
        return None
    return spec


def formula_to_vector(formula: str) -> np.ndarray:
    """
    Convert a molecular formula string to a fixed-length element-count vector.
    Elements tracked (indices 0-9): C, H, N, O, P, S, F, Cl, Br, I
    """
    import re
    elements = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
    vec = np.zeros(FORMULA_DIM, dtype=np.float32)
    for i, el in enumerate(elements):
        m = re.search(rf"{el}(\d*)", formula)
        if m:
            vec[i] = int(m.group(1)) if m.group(1) else 1
    return vec


################################################################
# ──────────────────────  Dataset  ────────────────────────────#
################################################################

class Data(Dataset):
    """
    Each item is a tuple:
        tokens       : list of Tensor[167]  — matched fragment vectors
        maccs        : Tensor[167]          — target MACCS bits
        formula_vecs : Tensor[K, 10]        — formula candidate vectors
                       K=1 at train (GT formula), K=MAX_CANDIDATES at infer

    When USE_FORMULA_ATTN is False, formula_vecs is a zero tensor (ignored).
    When the GT formula is missing from the MGF, we fall back to a zero vec
    (the attention module will learn to down-weight this).
    """

    def __init__(
        self,
        mgf_path: str,
        fp_bit_map_p_mode: dict,
        fp_bit_map_n_mode: dict,
        use_formula_attn: bool = USE_FORMULA_ATTN,
    ):
        path = Path(mgf_path)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {mgf_path}")

        spectra = list(load_from_mgf(str(path)))
        print(f"Loaded {len(spectra)} spectra from {path.name}")

        self.data      = []
        self.inchikeys = []
        skipped        = 0

        for spec in tqdm(spectra, desc=f"Preprocessing {path.name}"):
            spec   = preprocess(spec)
            smiles = spec.get(SMILES_KEY) if spec is not None else None
            if not smiles:
                skipped += 1
                continue
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                skipped += 1
                continue

            maccs = torch.tensor(
                np.array(MACCSkeys.GenMACCSKeys(mol)), dtype=torch.float32
            )
            inchikey = Chem.MolToInchiKey(mol) or ""
            self.inchikeys.append(inchikey[:14])

            # ── Formula candidates (K=1 at train: GT formula only) ──────────
            if use_formula_attn:
                formula = spec.get(FORMULA_KEY)
                if formula:
                    vec = formula_to_vector(str(formula))
                else:
                    vec = np.zeros(FORMULA_DIM, dtype=np.float32)
                formula_vecs = torch.tensor(vec[None, :], dtype=torch.float32)  # (1, 10)
            else:
                formula_vecs = torch.zeros(1, FORMULA_DIM, dtype=torch.float32)

            # ── Fragment token list ──────────────────────────────────────────
            mz_arr  = list(np.round(spec.peaks.mz, 2))
            int_arr = list(spec.peaks.intensities)
            pairs   = sorted(zip(mz_arr, int_arr), key=lambda x: x[1], reverse=True)

            tokens = []
            ionmode = spec.get("ionmode")
            bit_map = fp_bit_map_p_mode if ionmode == "positive" else (
                      fp_bit_map_n_mode if ionmode == "negative" else None)

            if bit_map is not None:
                for mz, _ in pairs:
                    if mz not in bit_map:
                        continue
                    vec = torch.tensor(bit_map[mz], dtype=torch.float32)
                    vec = vec / vec.max().clamp(min=1e-8)
                    tokens.append(vec)
                    if len(tokens) == MAX_FRAGMENTS:
                        break

            self.data.append((tokens, maccs, formula_vecs))

        print(f"  → {len(self.data)} valid  |  {skipped} skipped")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # (list[Tensor[167]], Tensor[167], Tensor[K,10])


def collate_fn(batch):
    token_lists = [item[0] for item in batch]
    targets     = torch.stack([item[1] for item in batch])          # (B, 167)
    max_k       = max(item[2].shape[0] for item in batch)
    formula_vecs = torch.zeros(len(batch), max_k, FORMULA_DIM)
    for b, item in enumerate(batch):
        k = item[2].shape[0]
        formula_vecs[b, :k] = item[2]
    return token_lists, targets, formula_vecs


################################################################
# ─────────────────────  Model  ───────────────────────────────#
################################################################

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FormulaEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FORMULA_DIM, d_model), nn.LayerNorm(d_model), Swish(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class FormulaAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale    = d_model ** -0.5
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, cls_emb, formula_emb, key_padding_mask=None):
        q      = self.q_proj(cls_emb).unsqueeze(1)
        k      = self.k_proj(formula_emb)
        v      = self.v_proj(formula_emb)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, v).squeeze(1)
        return self.out_proj(context), weights.squeeze(1)


class ProbablyWithFormulaAttn(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout,
                 use_formula_attn=USE_FORMULA_ATTN):
        super().__init__()
        self.use_formula_attn = use_formula_attn
        self.max_fragments    = MAX_FRAGMENTS

        self.input_proj = nn.Sequential(
            nn.Linear(MACCS_BITS, d_model), nn.LayerNorm(d_model), Swish(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))

        if use_formula_attn:
            self.formula_encoder = FormulaEncoder(d_model)
            self.formula_attn    = FormulaAttention(d_model)
            head_in_dim          = d_model * 2
        else:
            head_in_dim = d_model

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, d_model), Swish(),
            nn.Dropout(dropout), nn.Linear(d_model, MACCS_BITS),
        )

    def _encode_fragments(self, batch_vectors, device):
        B        = len(batch_vectors)
        tokens   = torch.zeros(B, self.max_fragments, MACCS_BITS, device=device)
        pad_mask = torch.ones(B, self.max_fragments + 1, dtype=torch.bool, device=device)
        pad_mask[:, 0] = False
        for b, vecs in enumerate(batch_vectors):
            n = min(len(vecs), self.max_fragments)
            if n > 0:
                tokens[b, :n] = torch.stack(vecs[:n]).to(device)
                pad_mask[b, 1:n + 1] = False
        frag = self.input_proj(tokens)
        cls  = self.cls_token.expand(B, -1, -1)
        x    = self.transformer(
            torch.cat([cls, frag], dim=1),
            src_key_padding_mask=pad_mask,
        )
        return x[:, 0, :]

    def forward(self, token_lists, formula_vecs):
        device  = self.cls_token.device
        cls_emb = self._encode_fragments(token_lists, device)

        if self.use_formula_attn:
            fv          = formula_vecs.to(device)
            pad_mask    = (fv.abs().sum(dim=-1) == 0)
            formula_emb = self.formula_encoder(fv)
            context, attn_weights = self.formula_attn(
                cls_emb, formula_emb, key_padding_mask=pad_mask
            )
            head_input = torch.cat([cls_emb, context], dim=-1)
        else:
            attn_weights = torch.zeros(cls_emb.shape[0], 1, device=device)
            head_input   = cls_emb

        return self.head(head_input), attn_weights


################################################################
# ─────────────────────  Inference wrapper  ───────────────────#
################################################################

class MS2MACCS:
    def __init__(
        self,
        bit_map_p_mode_path: str = "bit_map_p_mode.pkl",
        bit_map_n_mode_path: str = "bit_map_n_mode.pkl",
        model_path:          str = "model.pt",
        use_formula_attn:    bool = USE_FORMULA_ATTN,
        batch_size:          int  = 32,
        device:              str  = "cpu",
    ):
        self.use_formula_attn = use_formula_attn
        self.batch_size       = batch_size
        self.device           = torch.device(device)

        # ── Load model ──────────────────────────────────────────────────────
        self.model = ProbablyWithFormulaAttn(
            d_model=512,
            nhead=8,
            num_layers=4,
            dropout=0.1,
            use_formula_attn=use_formula_attn,
        )
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # ── Load fragment bit-maps ───────────────────────────────────────────
        with open(bit_map_p_mode_path, "rb") as f:
            self.bit_map_p_mode = pickle.load(f)
        with open(bit_map_n_mode_path, "rb") as f:
            self.bit_map_n_mode = pickle.load(f)

    def predict(self, mgf_path: str) -> torch.Tensor:
        """
        Run inference on all spectra in `mgf_path`.

        Returns
        -------
        preds : Tensor of shape (N, 167)
            Sigmoid-activated MACCS bit probabilities for every spectrum.
        """
        dataset = Data(
            mgf_path,
            self.bit_map_p_mode,
            self.bit_map_n_mode,
            use_formula_attn=self.use_formula_attn,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        all_preds = []
        with torch.no_grad():
            for token_lists, _targets, formula_vecs in tqdm(loader, desc="Predicting"):
                formula_vecs = formula_vecs.to(self.device)
                logits, _attn = self.model(token_lists, formula_vecs)  # (B, 167)
                probs = torch.sigmoid(logits)                           # (B, 167)
                all_preds.append(probs.cpu())

        return torch.cat(all_preds, dim=0)   # (N, 167)


################################################################
# ──────────────────────  Entry point  ────────────────────────#
################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MS2MACCS inference")
    parser.add_argument("mgf",              help="Path to input .mgf file")
    parser.add_argument("--model",          default="model.pt")
    parser.add_argument("--bit_map_p",      default="bit_map_p_mode.pkl")
    parser.add_argument("--bit_map_n",      default="bit_map_n_mode.pkl")
    parser.add_argument("--batch_size",     type=int,  default=32)
    parser.add_argument("--formula_attn",   action="store_true")
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--out",            default="predictions.pt",
                        help="Output path for saved predictions tensor")
    args = parser.parse_args()

    predictor = MS2MACCS(
        bit_map_p_mode_path=args.bit_map_p,
        bit_map_n_mode_path=args.bit_map_n,
        model_path=args.model,
        use_formula_attn=args.formula_attn,
        batch_size=args.batch_size,
        device=args.device,
    )

    preds = predictor.predict(args.mgf)          # (N, 167)
    torch.save(preds, args.out)
    print(f"Saved predictions {tuple(preds.shape)} → {args.out}")