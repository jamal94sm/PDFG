"""
PDFG Trainer  —  implements the full training loop from Section III.

Training procedure per iteration:
  1. Sample one mini-batch per source dataset
  2. Fourier-augment: generate x^{Di→Dj} for all pairs (i≠j)
  3. Forward pass:
       a. Original images → each through their own head  → L_sup
       b. Augmented images → through ALL heads            → L_sup (augmented)
       c. Compute L_ada (MK-MMD) between source feature sets
       d. Compute L_con (consistent loss)
       e. Compute L_d-t (dataset-aware triplet loss)
  4. Total loss = L_sup + L_ada + α·L_con + β·L_d-t
  5. Backward + optimizer step
"""

import time
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.feature_extractor import MultiDatasetExtractors
from losses.losses import PDFGLoss
from data.fourier_augment import fourier_augment_batch
from utils.metrics import evaluate


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _InfiniteLoader:
    """Wraps a DataLoader so iterating it never raises StopIteration."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter  = iter(loader)

    def next(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


def _sample_negative_augmented(
    aug_feats: torch.Tensor,   # [B, d]  augmented features
    labels: torch.Tensor,      # [B]     labels of augmented images
    anchor_labels: torch.Tensor,
) -> torch.Tensor:
    """
    For each anchor, pick an augmented feature from a DIFFERENT class.
    Falls back to a random sample if no strict negative exists.
    """
    B = anchor_labels.size(0)
    negatives = torch.zeros_like(aug_feats[:B])
    for i in range(B):
        mask = labels != anchor_labels[i]
        if mask.any():
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            chosen = idx[torch.randint(len(idx), (1,))]
            negatives[i] = aug_feats[chosen]
        else:
            negatives[i] = aug_feats[random.randint(0, len(aug_feats) - 1)]
    return negatives


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PDFGTrainer:
    """
    Full PDFG training pipeline.

    Args:
        model:                MultiDatasetExtractors
        loss_fn:              PDFGLoss
        source_loaders:       List of N DataLoaders (one per source spectrum)
        registration_loader:  Source test DataLoader (gallery for eval)
        target_loader:        Target test DataLoader (query for eval)
        device:               torch.device
        lr:                   Learning rate (1e-4 per paper)
        lam:                  Fourier augmentation λ (0.8 per paper)
        save_dir:             Directory to checkpoint best model
        eval_every:           Evaluate every N epochs
    """

    def __init__(
        self,
        model: MultiDatasetExtractors,
        loss_fn: PDFGLoss,
        source_loaders: List[DataLoader],
        registration_loader: DataLoader,
        target_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        lam: float = 0.8,
        save_dir: str = "checkpoints",
        eval_every: int = 5,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.source_loaders = source_loaders
        self.registration_loader = registration_loader
        self.target_loader = target_loader
        self.device = device
        self.lam = lam
        self.eval_every = eval_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer covers all model params + ArcFace classifier heads
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=lr,
        )

        self._inf_loaders = [_InfiniteLoader(ld) for ld in source_loaders]
        self.N = len(source_loaders)
        self.best_eer = float("inf")
        self.history = []

    # ------------------------------------------------------------------
    def _fetch_batches(self):
        """Sample one mini-batch from each source dataset."""
        batches = []
        for il in self._inf_loaders:
            imgs, labels = il.next()
            batches.append((imgs.to(self.device), labels.to(self.device)))
        return batches   # [(imgs_D0, lbl_D0), ..., (imgs_DN, lbl_DN)]

    # ------------------------------------------------------------------
    def _build_augmented(self, batches):
        """
        Fourier-augment all (Di → Dj) pairs where i ≠ j.
        Returns dict: (i, j) → (aug_imgs [B,C,H,W], orig_labels [B])
        """
        aug = {}
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    src_imgs, src_lbl = batches[i]
                    sty_imgs, _       = batches[j]
                    # Randomly pair style images if batch sizes differ
                    if sty_imgs.size(0) != src_imgs.size(0):
                        idx = torch.randint(sty_imgs.size(0), (src_imgs.size(0),))
                        sty_imgs = sty_imgs[idx]
                    aug_imgs = fourier_augment_batch(src_imgs, sty_imgs, self.lam)
                    aug[(i, j)] = (aug_imgs, src_lbl)   # same label as source
        return aug

    # ------------------------------------------------------------------
    def _train_step(self, batches, aug):
        """One gradient update step."""
        self.optimizer.zero_grad()

        N = self.N
        total_loss = torch.tensor(0.0, device=self.device)
        log = {"l_sup": 0., "l_ada": 0., "l_con": 0., "l_dt": 0.}

        # ── 1. Supervised loss on original images ──────────────────────
        orig_feats = []
        for i, (imgs, labels) in enumerate(batches):
            feat = self.model.extract(imgs, i)           # use head i
            l_sup = self.loss_fn.supervised_loss(feat, labels, i)
            total_loss += l_sup
            log["l_sup"] += l_sup.item()
            orig_feats.append(feat)

        # ── 2. Supervised + consistent loss on augmented images ────────
        # For each source Di, collect features of all (Di→Dj) images via all heads
        for i in range(N):
            aug_feats_all_pairs = []   # list over j≠i, each element = list-of-N-head-feats
            aug_labels_all = []

            for j in range(N):
                if i == j:
                    continue
                aug_imgs, aug_lbl = aug[(i, j)]

                # Supervised on augmented: through head i (same class space)
                feat_aug_i = self.model.extract(aug_imgs, i)
                l_sup_aug  = self.loss_fn.supervised_loss(feat_aug_i, aug_lbl, i)
                total_loss += l_sup_aug / (N - 1)
                log["l_sup"] += l_sup_aug.item() / (N - 1)

                # All-head features for consistent loss
                all_head_feats = self.model.extract_all(aug_imgs)  # list of N tensors [B,d]
                aug_feats_all_pairs.append(all_head_feats)
                aug_labels_all.append(aug_lbl)

            # ── 3. Consistent loss ─────────────────────────────────────
            l_con = self.loss_fn.consistent_loss(orig_feats[i], aug_feats_all_pairs)
            total_loss += l_con
            log["l_con"] += l_con.item()

            # ── 4. Dataset-aware triplet loss ──────────────────────────
            # Anchor = original source, Positive = same-class source,
            # Negative = diff-class augmented
            if len(aug_feats_all_pairs) > 0:
                # Average augmented features across heads and pairs for the negative pool
                aug_avg = torch.stack([
                    torch.stack(afs, dim=0).mean(0)
                    for afs in aug_feats_all_pairs
                ], dim=0).mean(0)              # [B, d]
                aug_neg_lbl = aug_labels_all[0]

                anchor   = orig_feats[i]
                positive = torch.roll(orig_feats[i], 1, dims=0)   # simple positive (same batch, shifted)
                negative = _sample_negative_augmented(aug_avg, aug_neg_lbl, batches[i][1])

                l_dt = self.loss_fn.triplet_loss(anchor, positive, negative)
                total_loss += l_dt
                log["l_dt"] += l_dt.item()

        # ── 5. Adaptation loss (MK-MMD) between every source pair ─────
        ada_count = 0
        for i in range(N):
            for j in range(i + 1, N):
                l_ada = self.loss_fn.adaptation_loss(orig_feats[i], orig_feats[j])
                total_loss += l_ada
                log["l_ada"] += l_ada.item()
                ada_count += 1
        if ada_count:
            log["l_ada"] /= ada_count

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        return {k: v / N for k, v in log.items()}, total_loss.item() / N

    # ------------------------------------------------------------------
    def train(self, epochs: int, steps_per_epoch: int = 100):
        """
        Full training loop.

        Args:
            epochs:           Number of training epochs
            steps_per_epoch:  Mini-batch steps per epoch (use len(smallest_loader))
        """
        print(f"\n{'═'*60}")
        print(f"  PDFG Training  |  {self.N} source spectra  |  {epochs} epochs")
        print(f"  Steps/epoch: {steps_per_epoch}  |  device: {self.device}")
        print(f"{'═'*60}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.loss_fn.train()
            epoch_log = {"l_sup": 0., "l_ada": 0., "l_con": 0., "l_dt": 0., "total": 0.}
            t0 = time.time()

            for step in range(steps_per_epoch):
                batches = self._fetch_batches()
                aug     = self._build_augmented(batches)
                log, total = self._train_step(batches, aug)

                for k in epoch_log:
                    if k != "total":
                        epoch_log[k] += log.get(k, 0.)
                epoch_log["total"] += total

            # Average over steps
            for k in epoch_log:
                epoch_log[k] /= steps_per_epoch

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"total={epoch_log['total']:.4f} | "
                f"sup={epoch_log['l_sup']:.4f} | "
                f"ada={epoch_log['l_ada']:.4f} | "
                f"con={epoch_log['l_con']:.4f} | "
                f"dt={epoch_log['l_dt']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Periodic evaluation
            if epoch % self.eval_every == 0:
                print(f"\n  ── Evaluation at epoch {epoch} ──")
                self.model.eval()
                acc, eer = evaluate(
                    self.model,
                    self.registration_loader,
                    self.target_loader,
                    self.device,
                    verbose=True,
                )
                self.history.append({"epoch": epoch, "acc": acc, "eer": eer, **epoch_log})

                # Save best checkpoint
                if eer < self.best_eer:
                    self.best_eer = eer
                    self._save("best_model.pth")
                    print(f"  ✓ New best EER: {eer:.2f}%  → saved best_model.pth")
                print()

        # Final save
        self._save("final_model.pth")
        print(f"\n  Training complete. Best EER: {self.best_eer:.2f}%")
        return self.history

    # ------------------------------------------------------------------
    def _save(self, filename: str):
        torch.save({
            "model_state":   self.model.state_dict(),
            "loss_fn_state": self.loss_fn.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "best_eer":      self.best_eer,
            "history":       self.history,
        }, self.save_dir / filename)

    def load(self, filename: str):
        ckpt = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.loss_fn.load_state_dict(ckpt["loss_fn_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_eer = ckpt["best_eer"]
        self.history  = ckpt.get("history", [])
        print(f"  Loaded checkpoint '{filename}'  (best EER: {self.best_eer:.2f}%)")
