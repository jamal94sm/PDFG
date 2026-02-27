"""
Evaluation utilities for CDPR-UT.

Paper uses two metrics:
  1. Identification accuracy  — query matched to nearest registration sample
  2. EER (Equal Error Rate)   — verification: genuine vs impostor matching
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model,                   # MultiDatasetExtractors
    loader: DataLoader,
    device: torch.device,
    mode: str = "average",   # "average" | "single"
    dataset_idx: int = 0,    # used only when mode="single"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract L2-normalized features from a DataLoader.

    Paper says: "features extracted by different feature extractors are
    averaged as the final feature and normalised by l2 normalization."

    Args:
        model:       MultiDatasetExtractors
        loader:      DataLoader to extract from
        device:      CUDA/CPU
        mode:        'average' → average over all N heads (paper's approach)
                     'single'  → use only head[dataset_idx]
        dataset_idx: used when mode='single'

    Returns:
        feats:  [N_samples, feature_dim]  float32 numpy array
        labels: [N_samples]               int64 numpy array
    """
    model.eval()
    all_feats, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        if mode == "average":
            feat_list = model.extract_all(imgs)                    # List of N [B, d]
            feats = torch.stack(feat_list, dim=0).mean(dim=0)     # [B, d]
        else:
            feats = model.extract(imgs, dataset_idx)               # [B, d]

        feats = F.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.vstack(all_feats), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
#  Identification accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_identification_accuracy(
    query_feats: np.ndarray,    # [Q, d]
    query_labels: np.ndarray,   # [Q]
    gallery_feats: np.ndarray,  # [G, d]
    gallery_labels: np.ndarray, # [G]
) -> float:
    """
    Nearest-neighbor identification accuracy.
    Each query is matched to the closest gallery sample (cosine similarity).

    Returns:
        accuracy in [0, 100] (percentage)
    """
    # Cosine similarity matrix [Q, G]
    Q = torch.from_numpy(query_feats)
    G = torch.from_numpy(gallery_feats)
    sim = torch.mm(Q, G.t())                       # [Q, G]
    pred_idx = sim.argmax(dim=1).numpy()           # [Q]
    predicted_labels = gallery_labels[pred_idx]   # [Q]
    acc = (predicted_labels == query_labels).mean() * 100.0
    return float(acc)


# ─────────────────────────────────────────────────────────────────────────────
#  EER for verification
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(
    query_feats: np.ndarray,    # [Q, d]
    query_labels: np.ndarray,   # [Q]
) -> float:
    """
    Compute EER from all pairwise matches within the query set.

    Genuine pairs:  same label → positive scores
    Impostor pairs: different label → negative scores

    Returns:
        EER in [0, 100] (percentage)
    """
    Q = torch.from_numpy(query_feats)
    sim_matrix = torch.mm(Q, Q.t()).numpy()  # [Q, Q]
    n = len(query_labels)

    genuine_scores, impostor_scores = [], []
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]
            if query_labels[i] == query_labels[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    genuine_scores  = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Sweep thresholds
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 500)

    min_diff = float("inf")
    eer = 0.0
    for thr in thresholds:
        far = (impostor_scores >= thr).mean()   # False Accept Rate
        frr = (genuine_scores  <  thr).mean()   # False Reject Rate
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0

    return float(eer * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    registration_loader: DataLoader,
    query_loader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Run the full CDPR-UT evaluation:
      - Registration set = source test images (gallery)
      - Query set        = target test images

    Returns:
        (accuracy %, EER %)
    """
    reg_feats,   reg_labels   = extract_features(model, registration_loader, device)
    query_feats, query_labels = extract_features(model, query_loader,        device)

    acc = compute_identification_accuracy(query_feats, query_labels, reg_feats, reg_labels)
    eer = compute_eer(query_feats, query_labels)

    if verbose:
        print(f"  Identification Accuracy : {acc:.2f}%")
        print(f"  Equal Error Rate (EER)  : {eer:.2f}%")

    return acc, eer
