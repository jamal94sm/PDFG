"""
CASIA Multi-Spectral Palmprint Dataset Loader.

Assumes ROI images are already extracted and organized as:
  root/
  ├── 460/
  │   ├── 001_l_460_01.jpg   (or any image format)
  │   ├── 001_l_460_02.jpg
  │   └── ...
  ├── 630/
  ├── 700/
  ├── 850/
  ├── 940/
  └── WHT/

Each filename should encode the subject/class ID so we can parse it.
We support two naming conventions:
  (A) Files directly under the spectrum folder, class parsed from filename prefix
      e.g.  001_l_460_01.jpg  → class "001"
  (B) Files organized into per-class subfolders
      root/460/001/image.jpg  → class "001"

Set `subfolder_mode=True` for convention (B).

Paper split: 3/4 training, 1/4 testing per class.
"""

import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image


CASIA_SPECTRA = ["460", "630", "700", "850", "940", "WHT"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ─────────────────────────────────────────────────────────────────────────────
#  Single-spectrum dataset
# ─────────────────────────────────────────────────────────────────────────────

class CASIASpectrum(Dataset):
    """
    One spectrum of the CASIA Multi-Spectral database.

    Args:
        root:           Path to the spectrum folder (e.g. .../460/)
        split:          'train' | 'test'  (3/4 train, 1/4 test per class)
        subfolder_mode: True  → class is the subfolder name
                        False → class is the numeric prefix in the filename
        transform:      torchvision transforms (default: resize 112×112, grayscale, normalize)
        seed:           Reproducibility seed for the train/test split
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        subfolder_mode: bool = False,
        transform=None,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.subfolder_mode = subfolder_mode
        self.transform = transform or self._default_transform()

        self._samples, self._label_map = self._build_samples(seed)

    # ------------------------------------------------------------------
    def _default_transform(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),   # → [-1, 1]
        ])

    # ------------------------------------------------------------------
    def _build_samples(self, seed: int) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
        """Walk root, collect (path, label) pairs, apply 3/4 : 1/4 split."""
        class_to_imgs: Dict[str, List[Path]] = {}

        if self.subfolder_mode:
            # Convention B: root/class_id/image.jpg
            for cls_dir in sorted(self.root.iterdir()):
                if not cls_dir.is_dir():
                    continue
                imgs = sorted(p for p in cls_dir.iterdir()
                              if p.suffix.lower() in IMG_EXTS)
                if imgs:
                    class_to_imgs[cls_dir.name] = imgs
        else:
            # Convention A: root/subject_xxx_...jpg
            # Parse leading digits as the class id
            for img_path in sorted(self.root.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                m = re.match(r"^(\d+)", img_path.name)
                cls_id = m.group(1) if m else img_path.stem
                class_to_imgs.setdefault(cls_id, []).append(img_path)

        # Build integer label map
        label_map = {cls: idx for idx, cls in enumerate(sorted(class_to_imgs))}

        samples: List[Tuple[str, int]] = []
        for cls_name, imgs in class_to_imgs.items():
            label = label_map[cls_name]
            rng = random.Random(seed + label)
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
        img = Image.open(path).convert("L")   # grayscale
        return self.transform(img), label

    def get_label_map(self) -> Dict[str, int]:
        return dict(self._label_map)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-spectrum manager  (core interface for training)
# ─────────────────────────────────────────────────────────────────────────────

class CASIAMultiSpectral:
    """
    Manages source + target spectra for one CDPR-UT experiment.

    Paper experiments use 2 or 3 source spectra + 1 target spectrum.

    Args:
        root:           Root directory containing the spectrum subfolders
        source_spectra: List of spectrum names to use for training
                        e.g. ["460", "630", "700"]
        target_spectrum:Spectrum name held out as unseen target
                        e.g. "850"
        subfolder_mode: True if images are in per-class subfolders
        batch_size:     Mini-batch size (8 per paper)
        num_workers:    DataLoader workers
        seed:           Reproducibility seed
    """

    def __init__(
        self,
        root: str,
        source_spectra: List[str],
        target_spectrum: str,
        subfolder_mode: bool = False,
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.source_spectra = source_spectra
        self.target_spectrum = target_spectrum
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Build source train/test datasets
        self.src_train: List[CASIASpectrum] = [
            CASIASpectrum(self.root / s, "train", subfolder_mode, seed=seed)
            for s in source_spectra
        ]
        self.src_test: List[CASIASpectrum] = [
            CASIASpectrum(self.root / s, "test", subfolder_mode, seed=seed)
            for s in source_spectra
        ]

        # Target is test-only (unseen during training)
        self.tgt_test = CASIASpectrum(
            self.root / target_spectrum, "test", subfolder_mode, seed=seed
        )

        self.num_classes_per_src = [ds.num_classes for ds in self.src_train]

    # ------------------------------------------------------------------
    #  DataLoaders
    # ------------------------------------------------------------------

    def source_train_loaders(self) -> List[DataLoader]:
        """One shuffled DataLoader per source spectrum."""
        return [
            DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                       num_workers=self.num_workers, drop_last=True,
                       pin_memory=True)
            for ds in self.src_train
        ]

    def source_test_loaders(self) -> List[DataLoader]:
        """One ordered DataLoader per source spectrum (for registration set)."""
        return [
            DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                       num_workers=self.num_workers, pin_memory=True)
            for ds in self.src_test
        ]

    def target_loader(self) -> DataLoader:
        """DataLoader for the unseen target spectrum (query set)."""
        return DataLoader(
            self.tgt_test, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

    def registration_loader(self) -> DataLoader:
        """All source test samples combined (for identification evaluation)."""
        combined = ConcatDataset(self.src_test)
        return DataLoader(combined, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    # ------------------------------------------------------------------
    def summary(self):
        w = 55
        print(f"\n{'─'*w}")
        print(f"  CASIA Multi-Spectral  |  CDPR-UT Experiment")
        print(f"{'─'*w}")
        print(f"  {'Spectrum':<10} {'Role':<14} {'Classes':>7} {'Train imgs':>10} {'Test imgs':>10}")
        print(f"  {'─'*8:<10} {'─'*12:<14} {'─'*7:>7} {'─'*10:>10} {'─'*10:>10}")
        for name, tr, te in zip(self.source_spectra, self.src_train, self.src_test):
            print(f"  {name:<10} {'Source':<14} {tr.num_classes:>7} {len(tr):>10} {len(te):>10}")
        tgt = self.tgt_test
        print(f"  {self.target_spectrum:<10} {'Target (unseen)':<14} {tgt.num_classes:>7} {'─':>10} {len(tgt):>10}")
        print(f"{'─'*w}\n")
