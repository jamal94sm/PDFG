"""
CASIA Multi-Spectral Palmprint Dataset for PDFG.

Written to match the exact data structure in your existing codebase:

  Data path : /home/pai-ng/Jamal/CASIA-MS-ROI   (flat folder, no subfolders)
  File name : {subject_id}_{hand}_{spectrum}_{iteration}.jpg
  Example   : 001_l_460_01.jpg

  Identity (label) : subject_id + hand  →  e.g. "001_l"  (one palm = one class)
  Domain           : spectrum string    →  e.g. "460", "630", "700", "850", "940", "WHT"

  Image mode : RGB  (matches your .convert("RGB"))
  Image size : 224×224  (matches your existing resize)

Split: 3/4 training, 1/4 testing per identity, consistent with the PDFG paper.
"""

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image


CASIA_SPECTRA = ["460", "630", "700", "850", "940", "WHT"]


# ─────────────────────────────────────────────────────────────────────────────
#  Single-spectrum dataset
# ─────────────────────────────────────────────────────────────────────────────

class CASIASpectrum(Dataset):
    """
    All images belonging to one spectrum, loaded from a flat directory.

    Filename convention (from your code):
        {subject_id}_{hand}_{spectrum}_{iteration}.jpg
        e.g.  001_l_460_01.jpg

    Identity label = subject_id + "_" + hand   →  "001_l"
    This matches your hand_id = f"{subject_id}_{hand}" logic exactly.

    Args:
        data_path : root folder containing all .jpg files (flat, no subfolders)
        spectrum  : which spectrum to load, e.g. "460"
        split     : "train" (3/4) or "test" (1/4)
        transform : torchvision transform; defaults to 224×224 RGB + ToTensor
        seed      : reproducibility seed for the train/test split
    """

    def __init__(
        self,
        data_path: str,
        spectrum: str,
        split: str = "train",
        transform=None,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.spectrum  = spectrum
        self.split     = split
        self.transform = transform or self._default_transform()

        self._samples, self._label_map = self._build_samples(seed)

    # ------------------------------------------------------------------
    def _default_transform(self):
        """224×224 RGB — matches your existing to_tensor in CASIA_MS_Dataset."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),           # → [0, 1]
        ])

    # ------------------------------------------------------------------
    def _build_samples(self, seed: int) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
        """
        Walk the flat data_path, keep only files for self.spectrum,
        parse hand_id from filename, apply 3/4:1/4 train/test split.
        """
        # Collect all files for this spectrum, grouped by hand_id
        class_to_imgs: Dict[str, List[Path]] = {}

        for fpath in sorted(self.data_path.iterdir()):
            if not fpath.suffix.lower() == ".jpg":
                continue
            # Parse: subject_id _ hand _ spectrum _ iteration
            parts = fpath.stem.split("_")
            if len(parts) != 4:
                continue
            subject_id, hand, spectrum, iteration = parts
            if spectrum != self.spectrum:
                continue

            hand_id = f"{subject_id}_{hand}"   # matches your code exactly
            class_to_imgs.setdefault(hand_id, []).append(fpath)

        # Build integer label map (sorted for reproducibility)
        label_map: Dict[str, int] = {
            hid: idx for idx, hid in enumerate(sorted(class_to_imgs))
        }

        # Apply 3/4 train : 1/4 test split per identity
        samples: List[Tuple[str, int]] = []
        for hand_id, imgs in class_to_imgs.items():
            label = label_map[hand_id]
            rng   = random.Random(seed + label)
            imgs_shuffled = list(imgs)
            rng.shuffle(imgs_shuffled)
            split_idx = max(1, int(len(imgs_shuffled) * 0.75))
            chosen = imgs_shuffled[:split_idx] if self.split == "train" else imgs_shuffled[split_idx:]
            samples.extend((str(p), label) for p in chosen)

        return samples, label_map

    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self._label_map)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self._samples[idx]
        img = Image.open(path).convert("RGB")   # matches your .convert("RGB")
        return self.transform(img), label

    def get_label_map(self) -> Dict[str, int]:
        return dict(self._label_map)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-spectrum manager
# ─────────────────────────────────────────────────────────────────────────────

class CASIAMultiSpectral:
    """
    Manages N source spectra + 1 unseen target spectrum for CDPR-UT.

    Usage example (matching your train_domains / test_domains):
        dm = CASIAMultiSpectral(
            data_path       = "/home/pai-ng/Jamal/CASIA-MS-ROI",
            source_spectra  = ["460", "700", "630"],   # your train_domains
            target_spectrum = "940",                   # your test_domains[0]
        )

    Args:
        data_path       : flat folder with all .jpg files
        source_spectra  : list of spectrum names to use for training
        target_spectrum : spectrum held out as unseen target
        batch_size      : DataLoader batch size (8 per PDFG paper)
        num_workers     : DataLoader workers
        seed            : reproducibility seed
    """

    def __init__(
        self,
        data_path: str,
        source_spectra: List[str],
        target_spectrum: str,
        batch_size: int = 8,
        num_workers: int = 2,
        seed: int = 42,
    ):
        self.data_path       = data_path
        self.source_spectra  = source_spectra
        self.target_spectrum = target_spectrum
        self.batch_size      = batch_size
        self.num_workers     = num_workers

        # Source: train + test splits
        self.src_train: List[CASIASpectrum] = [
            CASIASpectrum(data_path, s, "train", seed=seed)
            for s in source_spectra
        ]
        self.src_test: List[CASIASpectrum] = [
            CASIASpectrum(data_path, s, "test", seed=seed)
            for s in source_spectra
        ]

        # Target: test only (never seen during training)
        self.tgt_test = CASIASpectrum(data_path, target_spectrum, "test", seed=seed)

        # How many identity classes per source dataset (needed for ArcFace heads)
        self.num_classes_per_src: List[int] = [ds.num_classes for ds in self.src_train]

    # ------------------------------------------------------------------
    def source_train_loaders(self) -> List[DataLoader]:
        """One shuffled DataLoader per source spectrum (for training)."""
        return [
            DataLoader(
                ds,
                batch_size  = self.batch_size,
                shuffle     = True,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = True,
            )
            for ds in self.src_train
        ]

    def source_test_loaders(self) -> List[DataLoader]:
        """One ordered DataLoader per source spectrum."""
        return [
            DataLoader(
                ds,
                batch_size  = self.batch_size,
                shuffle     = False,
                num_workers = self.num_workers,
                pin_memory  = True,
            )
            for ds in self.src_test
        ]

    def target_loader(self) -> DataLoader:
        """DataLoader for the unseen target spectrum (query set)."""
        return DataLoader(
            self.tgt_test,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

    def registration_loader(self) -> DataLoader:
        """All source test samples combined (gallery for identification)."""
        return DataLoader(
            ConcatDataset(self.src_test),
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

    # ------------------------------------------------------------------
    def summary(self):
        w = 60
        print(f"\n{'─'*w}")
        print(f"  CASIA Multi-Spectral  |  CDPR-UT")
        print(f"{'─'*w}")
        print(f"  {'Spectrum':<10} {'Role':<18} {'Classes':>7} {'Train':>8} {'Test':>8}")
        print(f"  {'─'*8:<10} {'─'*16:<18} {'─'*7:>7} {'─'*8:>8} {'─'*8:>8}")
        for name, tr, te in zip(self.source_spectra, self.src_train, self.src_test):
            print(f"  {name:<10} {'Source':<18} {tr.num_classes:>7} {len(tr):>8} {len(te):>8}")
        tgt = self.tgt_test
        print(f"  {self.target_spectrum:<10} {'Target (unseen)':<18} {tgt.num_classes:>7} {'—':>8} {len(tgt):>8}")
        print(f"{'─'*w}\n")
