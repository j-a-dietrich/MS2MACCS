
import torch

def calc_tanimoto(pred_maccs, true_maccs):
    """calculates the tanimoto score for a batch of two tensors"""

    intersection = (pred_maccs * true_maccs).sum(dim=1)
    union = (pred_maccs + true_maccs).clamp(max=1).sum(dim=1)
    tanimoto = torch.where(union > 0, intersection / union, torch.ones_like(union))

    return tanimoto.mean().item()


def evaluate(model, loader, loss_fn, DEVICE="cuda"):
    model.eval()
    total_loss = 0.0
    total_pred = []
    total_true = []

    with torch.no_grad():
        for submaccs, maccs in loader: 
            maccs = maccs.to(DEVICE)
            logits = model(submaccs)

            loss = loss_fn(logits, maccs)
            total_loss += loss.item() * len(maccs)
            pred_maccs = (torch.sigmoid(logits) >= 0.5).float()
            total_pred.append(pred_maccs.cpu())
            total_true.append(maccs.cpu())

        avg_loss = total_loss / len(loader.dataset)
        tanimoto = calc_tanimoto(torch.cat(total_pred), torch.cat(total_true))
        
        return avg_loss, tanimoto