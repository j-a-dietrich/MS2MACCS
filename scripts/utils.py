
import torch

def calc_tanimoto(pred_maccs, true_maccs):
    """calculates the tanimoto score for a batch of two tensors"""

    intersection = (pred_maccs * true_maccs).sum(dim=1)
    union = (pred_maccs + true_maccs).clamp(max=1).sum(dim=1)
    tanimoto = torch.where(union > 0, intersection / union, torch.ones_like(union))

    return tanimoto.mean().item()