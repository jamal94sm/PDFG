"""
Loss functions for PDFG method (Section III-D of the paper):

  L = L_sup + L_ada + α·L_con + β·L_d-t
    = L_ArcFace + L_MK-MMD + α·L_con + β·L_d-t

References:
  - ArcFace:   Deng et al., IEEE TPAMI 2022
  - MK-MMD:    Gretton et al., NeurIPS 2012
  - L_con:     Eq. (9) in the paper
  - L_d-t:     Eq. (10) in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. ArcFace Loss (Supervised Loss L_sup)  — Eq. (1)
# ---------------------------------------------------------------------------

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.

    Args:
        in_features:  Dimensionality of input features
        num_classes:  Number of identities (classes) in the source dataset
        s:            Feature scale (default 64)
        m:            Angular margin in radians (default 0.5 as in paper)
    """

    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)   # cos(π - m) threshold
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weights and features → cosine similarity
        cosine = F.linear(features, F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        # For numerical stability: if cos θ < cos(π − m), use the safe approximation
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only on the ground-truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return F.cross_entropy(output, labels)


# ---------------------------------------------------------------------------
# 2. MK-MMD Loss (Domain Adaptation Loss L_ada)  — Eq. (8)
# ---------------------------------------------------------------------------

class MKMMDLoss(nn.Module):
    """
    Multiple Kernel Maximum Mean Discrepancy (MK-MMD).
    Measures distribution distance between two feature sets in RKHS.

    Args:
        kernels: List of bandwidth values for Gaussian kernels.
                 Default follows the standard multi-scale setting.
    """

    def __init__(self, kernels=(1, 5, 10, 20, 50, 100)):
        super().__init__()
        self.kernels = kernels

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """k(x, y) = exp(−‖x − y‖² / bandwidth)"""
        dist = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-dist / bandwidth)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: Features from dataset D1  [A, d]
            target: Features from dataset D2  [B, d]
        Returns:
            Scalar MK-MMD loss
        """
        n_s = source.size(0)
        n_t = target.size(0)

        loss = torch.tensor(0.0, device=source.device)
        for bw in self.kernels:
            k_ss = self._gaussian_kernel(source, source, bw).sum() / (n_s * n_s)
            k_st = self._gaussian_kernel(source, target, bw).sum() / (n_s * n_t)
            k_tt = self._gaussian_kernel(target, target, bw).sum() / (n_t * n_t)
            loss += k_ss - 2 * k_st + k_tt

        return loss / len(self.kernels)


# ---------------------------------------------------------------------------
# 3. Consistent Loss (L_con)  — Eq. (9)
# ---------------------------------------------------------------------------

class ConsistentLoss(nn.Module):
    """
    Encourages augmented features (from all N heads) to be consistent
    with the original source features.

    For image x^Dj_i with original feature f(x^Dj_i)^j,
    N-1 augmented versions x^{Dj→Dn}_i are generated.
    Each is passed through ALL N extractors; their average should
    match the original feature.

    L_con = Σ_{n≠j} ‖ f(x^Dj_i)^j  −  (1/N) Σ_l f(x^{Dj→Dn}_i)^l ‖²
    """

    def forward(
        self,
        original_feat: torch.Tensor,          # [B, d]  f(x^Dj)^j
        augmented_feats_list: list,           # List of [B, d] — one per augmented image,
                                              # each element is itself a list over N heads
    ) -> torch.Tensor:
        """
        Args:
            original_feat:       Features of original images via their own head. [B, d]
            augmented_feats_list: For each augmented image (N-1 of them),
                                  a list of N tensors [B, d] (one per head).
        """
        loss = torch.tensor(0.0, device=original_feat.device)
        count = 0

        for aug_feats in augmented_feats_list:
            # aug_feats: list of N tensors [B, d]
            avg_feat = torch.stack(aug_feats, dim=0).mean(dim=0)  # [B, d]
            loss += F.mse_loss(original_feat, avg_feat)
            count += 1

        return loss / max(count, 1)


# ---------------------------------------------------------------------------
# 4. Dataset-Aware Triplet Loss (L_d-t)  — Eq. (10)
# ---------------------------------------------------------------------------

class DatasetAwareTripletLoss(nn.Module):
    """
    Triplet loss where:
      - anchor   = source original image feature
      - positive = another source image of the SAME class
      - negative = an augmented image of a DIFFERENT class

    L_d-t = max(0,  D_ap  −  D_c-an  +  t)

    Args:
        margin (t): Default 0.4 as specified in paper.
    """

    def __init__(self, margin: float = 0.4):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,    # [B, d] source features (anchor)
        positive: torch.Tensor,  # [B, d] same-class source features
        negative: torch.Tensor,  # [B, d] different-class augmented features
    ) -> torch.Tensor:
        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()


# ---------------------------------------------------------------------------
# 5. Combined PDFG Loss
# ---------------------------------------------------------------------------

class PDFGLoss(nn.Module):
    """
    Full combined loss from Eq. (11):
      L = L_ArcFace + L_MK-MMD + α·L_con + β·L_d-t

    Args:
        num_classes_per_dataset: List of class counts, one per source dataset
        feature_dim:             Feature dimensionality
        alpha:                   Weight for L_con  (default 0.1 per paper)
        beta:                    Weight for L_d-t  (default 1.0 per paper)
        s:                       ArcFace scale
        m:                       ArcFace angular margin (0.5 per paper)
    """

    def __init__(
        self,
        num_classes_per_dataset: list,
        feature_dim: int = 128,
        alpha: float = 0.1,
        beta: float = 1.0,
        s: float = 64.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.arcface_heads = nn.ModuleList([
            ArcFaceLoss(feature_dim, n_cls, s=s, m=m)
            for n_cls in num_classes_per_dataset
        ])
        self.mkmmd = MKMMDLoss()
        self.consistent = ConsistentLoss()
        self.triplet = DatasetAwareTripletLoss()

    def supervised_loss(
        self, features: torch.Tensor, labels: torch.Tensor, dataset_idx: int
    ) -> torch.Tensor:
        return self.arcface_heads[dataset_idx](features, labels)

    def adaptation_loss(
        self, feats_d1: torch.Tensor, feats_d2: torch.Tensor
    ) -> torch.Tensor:
        return self.mkmmd(feats_d1, feats_d2)

    def consistent_loss(
        self,
        original_feat: torch.Tensor,
        augmented_feats_list: list,
    ) -> torch.Tensor:
        return self.consistent(original_feat, augmented_feats_list)

    def triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.triplet(anchor, positive, negative)

    def forward(
        self,
        # Supervised
        src_feats: torch.Tensor,
        src_labels: torch.Tensor,
        dataset_idx: int,
        # Supervised on augmented
        aug_feats: torch.Tensor,
        aug_labels: torch.Tensor,
        aug_dataset_idx: int,
        # Adaptation
        feats_d1: torch.Tensor,
        feats_d2: torch.Tensor,
        # Consistent
        original_feat: torch.Tensor,
        augmented_feats_list: list,
        # Triplet
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> dict:
        l_sup_src = self.supervised_loss(src_feats, src_labels, dataset_idx)
        l_sup_aug = self.supervised_loss(aug_feats, aug_labels, aug_dataset_idx)
        l_sup = l_sup_src + l_sup_aug

        l_ada = self.adaptation_loss(feats_d1, feats_d2)
        l_con = self.consistent_loss(original_feat, augmented_feats_list)
        l_dt  = self.triplet_loss(anchor, positive, negative)

        total = l_sup + l_ada + self.alpha * l_con + self.beta * l_dt

        return {
            "total": total,
            "l_sup": l_sup,
            "l_ada": l_ada,
            "l_con": l_con,
            "l_dt":  l_dt,
        }
