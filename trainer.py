"""
PDFG Trainer  —  two-phase training exactly as described in Section III.

The paper states (Section III-B, emphasis added):
  "The feature extractors are FIRSTLY TRAINED by L_sup using the
   LABELED SOURCE IMAGES AND AUGMENTED IMAGES."

And from Fig. 2 / Section III:
  "N feature extractors that have specific fully connected layers while
   SHARE THE PREVIOUS LAYERS are constructed. For each dataset, the
   feature extractor is firstly trained by supervised loss USING THEIR LABELS."

So the correct reading is:

  PHASE 1  (supervised pre-training, L_sup only)
  ─────────────────────────────────────────────
  • N heads, one per source dataset (paper's architecture, unchanged)
  • Each head trained with ArcFace on ITS OWN dataset's label space
  • Input = original source images  +  Fourier-augmented images
    (augmented x^{Di→Dj} carries the same label as x^{Di}, so it feeds
     into head i with the same labels — same head, same class space)
  • Shared CNN layers get gradients from all N heads simultaneously

  PHASE 2  (full PDFG, all losses, Eq. 11)
  ─────────────────────────────────────────
  • Same N heads, now additionally trained with L_ada + L_con + L_d-t
  • Fourier augmentation continues to be used
"""

import time
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from models.feature_extractor import MultiDatasetExtractors
from losses.losses import PDFGLoss
from data.fourier_augment import fourier_augment_batch
from utils.metrics import evaluate


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _InfiniteLoader:
    """Wraps a DataLoader so it never raises StopIteration."""

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
    aug_feats: torch.Tensor,
    aug_labels: torch.Tensor,
    anchor_labels: torch.Tensor,
) -> torch.Tensor:
    """For each anchor pick an augmented feature from a different class."""
    B = anchor_labels.size(0)
    negatives = torch.zeros_like(aug_feats[:B])
    for i in range(B):
        mask = aug_labels != anchor_labels[i]
        if mask.any():
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            negatives[i] = aug_feats[idx[torch.randint(len(idx), (1,))]]
        else:
            negatives[i] = aug_feats[random.randint(0, B - 1)]
    return negatives


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PDFGTrainer:
    """
    Two-phase PDFG training.

    Phase 1 — supervised pre-training (L_sup only):
        Each of the N feature extractors is trained with ArcFace on its
        own source dataset's labels. Inputs are BOTH original source images
        AND Fourier-augmented images x^{Di→Dj} — augmented images use the
        same labels as their Di source, fed through head i.
        The shared CNN backbone learns from all N heads simultaneously.

    Phase 2 — full PDFG (L_sup + L_ada + alpha*L_con + beta*L_d-t):
        Continues from pre-trained weights. Adds cross-dataset adaptation,
        consistent loss, and dataset-aware triplet loss.
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
        pretrain_epochs: int = 30,
    ):
        self.model    = model.to(device)
        self.loss_fn  = loss_fn.to(device)
        self.source_loaders      = source_loaders
        self.registration_loader = registration_loader
        self.target_loader       = target_loader
        self.device          = device
        self.lam             = lam
        self.eval_every      = eval_every
        self.pretrain_epochs = pretrain_epochs
        self.save_dir        = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.N = len(source_loaders)
        self._inf_loaders = [_InfiniteLoader(ld) for ld in source_loaders]

        # Single optimizer for both phases — Phase 1 simply doesn't
        # activate the adaptation/consistent/triplet losses.
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=lr,
        )

        self.best_eer = float("inf")
        self.history  = []

    # ──────────────────────────────────────────────────────────────────
    #  Shared helpers
    # ──────────────────────────────────────────────────────────────────

    def _fetch_batches(self) -> list:
        """Sample one mini-batch from each source dataset."""
        return [
            (imgs.to(self.device), labels.to(self.device))
            for imgs, labels in (il.next() for il in self._inf_loaders)
        ]

    def _build_augmented(self, batches: list) -> dict:
        """
        Fourier-augment all (Di -> Dj) pairs (i != j).
        Returns {(i,j): (aug_imgs, src_labels_i)}.
        aug_imgs carry the same identity as Di.
        """
        aug = {}
        for i in range(self.N):
            src_imgs, src_lbl = batches[i]
            for j in range(self.N):
                if i == j:
                    continue
                sty_imgs, _ = batches[j]
                if sty_imgs.size(0) != src_imgs.size(0):
                    idx = torch.randint(sty_imgs.size(0), (src_imgs.size(0),))
                    sty_imgs = sty_imgs[idx]
                aug[(i, j)] = (fourier_augment_batch(src_imgs, sty_imgs, self.lam), src_lbl)
        return aug

    # ──────────────────────────────────────────────────────────────────
    #  PHASE 1 — supervised pre-training step
    # ──────────────────────────────────────────────────────────────────

    def _pretrain_step(self, batches: list, aug: dict) -> float:
        """
        L_sup on original + augmented images using N heads.

        For each source Di:
          - x^Di           → head i → ArcFace with labels_i
          - x^{Di->Dj}     → head i → ArcFace with same labels_i
            (augmented images have same identity, just new style)
        """
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device)

        for i, (src_imgs, src_lbl) in enumerate(batches):
            # Original source images through head i
            feat_orig = self.model.extract(src_imgs, i)
            total_loss += self.loss_fn.supervised_loss(feat_orig, src_lbl, i)

            # Augmented images x^{Di->Dj} for all j != i, also through head i
            for j in range(self.N):
                if i == j:
                    continue
                aug_imgs, aug_lbl = aug[(i, j)]
                feat_aug = self.model.extract(aug_imgs, i)
                total_loss += self.loss_fn.supervised_loss(feat_aug, aug_lbl, i)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return total_loss.item()

    # ──────────────────────────────────────────────────────────────────
    #  PHASE 2 — full PDFG step (Eq. 11)
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batches: list, aug: dict) -> tuple:
        """L = L_sup + L_ada + alpha*L_con + beta*L_d-t"""
        self.optimizer.zero_grad()

        N = self.N
        total_loss = torch.tensor(0.0, device=self.device)
        log = {"l_sup": 0., "l_ada": 0., "l_con": 0., "l_dt": 0.}

        # ── 1. L_sup: original + augmented images via their own head ───
        orig_feats = []
        for i, (src_imgs, src_lbl) in enumerate(batches):
            feat_orig = self.model.extract(src_imgs, i)
            l_sup = self.loss_fn.supervised_loss(feat_orig, src_lbl, i)
            total_loss += l_sup
            log["l_sup"] += l_sup.item()
            orig_feats.append(feat_orig)

            for j in range(N):
                if i == j:
                    continue
                aug_imgs, aug_lbl = aug[(i, j)]
                feat_aug  = self.model.extract(aug_imgs, i)
                l_sup_aug = self.loss_fn.supervised_loss(feat_aug, aug_lbl, i)
                total_loss += l_sup_aug / (N - 1)
                log["l_sup"] += l_sup_aug.item() / (N - 1)

        # ── 2. L_con + L_d-t (per source dataset) ─────────────────────
        for i in range(N):
            aug_feats_all_pairs = []
            aug_labels_all      = []

            for j in range(N):
                if i == j:
                    continue
                aug_imgs, aug_lbl = aug[(i, j)]
                # Pass through ALL N heads (needed for consistent loss)
                all_head_feats = self.model.extract_all(aug_imgs)
                aug_feats_all_pairs.append(all_head_feats)
                aug_labels_all.append(aug_lbl)

            # L_con: f(x^Di)^i should match the mean of augmented features
            l_con = self.loss_fn.consistent_loss(orig_feats[i], aug_feats_all_pairs)
            total_loss += l_con
            log["l_con"] += l_con.item()

            # L_d-t: anchor=orig, positive=rolled orig, negative=diff-class aug
            if aug_feats_all_pairs:
                aug_avg = torch.stack([
                    torch.stack(hf, dim=0).mean(0)
                    for hf in aug_feats_all_pairs
                ], dim=0).mean(0)

                anchor   = orig_feats[i]
                positive = torch.roll(orig_feats[i], 1, dims=0)
                negative = _sample_negative_augmented(
                    aug_avg, aug_labels_all[0], batches[i][1]
                )
                l_dt = self.loss_fn.triplet_loss(anchor, positive, negative)
                total_loss += l_dt
                log["l_dt"] += l_dt.item()

        # ── 3. L_ada: MK-MMD between every pair of source datasets ────
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

    # ──────────────────────────────────────────────────────────────────
    #  Public entry point
    # ──────────────────────────────────────────────────────────────────

    def train(self, epochs: int, steps_per_epoch: int = 100):
        """Run Phase 1 (supervised pre-train) then Phase 2 (full PDFG)."""
        self._run_phase1(steps_per_epoch)
        self._run_phase2(epochs, steps_per_epoch)
        return self.history

    # ──────────────────────────────────────────────────────────────────
    #  Phase 1 loop
    # ──────────────────────────────────────────────────────────────────

    def _run_phase1(self, steps_per_epoch: int):
        print(f"\n{'='*60}")
        print(f"  PHASE 1 — Supervised Pre-training  (L_sup only)")
        print(f"  Input: original + Fourier-augmented source images")
        print(f"  N heads, each trained on its own dataset's labels")
        print(f"  Epochs: {self.pretrain_epochs}  |  Steps/epoch: {steps_per_epoch}")
        print(f"{'='*60}")

        for epoch in range(1, self.pretrain_epochs + 1):
            self.model.train()
            self.loss_fn.train()
            epoch_loss = 0.0
            t0 = time.time()

            for _ in range(steps_per_epoch):
                batches    = self._fetch_batches()
                aug        = self._build_augmented(batches)
                epoch_loss += self._pretrain_step(batches, aug)

            epoch_loss /= steps_per_epoch
            print(f"  Epoch {epoch:3d}/{self.pretrain_epochs} | "
                  f"L_sup={epoch_loss:.4f} | {time.time()-t0:.1f}s")

        self._save("pretrained_model.pth")
        print(f"\n  Phase 1 complete -> pretrained_model.pth\n")

    # ──────────────────────────────────────────────────────────────────
    #  Phase 2 loop
    # ──────────────────────────────────────────────────────────────────

    def _run_phase2(self, epochs: int, steps_per_epoch: int):
        print(f"{'='*60}")
        print(f"  PHASE 2 — Full PDFG Training  (all losses, Eq. 11)")
        print(f"  Epochs: {epochs}  |  Steps/epoch: {steps_per_epoch}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.loss_fn.train()
            elog = {"l_sup": 0., "l_ada": 0., "l_con": 0., "l_dt": 0., "total": 0.}
            t0 = time.time()

            for _ in range(steps_per_epoch):
                batches    = self._fetch_batches()
                aug        = self._build_augmented(batches)
                log, total = self._train_step(batches, aug)
                for k in log:
                    elog[k] += log[k]
                elog["total"] += total

            for k in elog:
                elog[k] /= steps_per_epoch

            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"total={elog['total']:.4f} | "
                f"sup={elog['l_sup']:.4f} | "
                f"ada={elog['l_ada']:.4f} | "
                f"con={elog['l_con']:.4f} | "
                f"dt={elog['l_dt']:.4f} | "
                f"{time.time()-t0:.1f}s"
            )

            if epoch % self.eval_every == 0:
                print(f"\n  -- Evaluation at epoch {epoch} --")
                self.model.eval()
                acc, eer = evaluate(
                    self.model, self.registration_loader,
                    self.target_loader, self.device, verbose=True,
                )
                self.history.append(
                    {"phase": 2, "epoch": epoch, "acc": acc, "eer": eer, **elog}
                )
                if eer < self.best_eer:
                    self.best_eer = eer
                    self._save("best_model.pth")
                    print(f"  New best EER: {eer:.2f}%  -> best_model.pth")
                print()

        self._save("final_model.pth")
        print(f"\n  Training complete. Best EER: {self.best_eer:.2f}%")

    # ──────────────────────────────────────────────────────────────────
    #  Checkpointing
    # ──────────────────────────────────────────────────────────────────

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
        self.best_eer = ckpt.get("best_eer", float("inf"))
        self.history  = ckpt.get("history", [])
        print(f"  Loaded '{filename}'  (best EER: {self.best_eer:.2f}%)")
