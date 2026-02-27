# PDFG — Palmprint Data and Feature Generation

PyTorch implementation of:

> **"Learning to Generalize Unseen Dataset for Cross-Dataset Palmprint Recognition"**  
> Shao et al., IEEE Transactions on Information Forensics and Security, Vol. 19, 2024  
> DOI: 10.1109/TIFS.2024.3371257

---

## Overview

PDFG tackles **CDPR-UT** (Cross-Dataset Palmprint Recognition with **Unseen** Target dataset) — the most realistic but hardest scenario where the target dataset is completely unavailable during training.

| Scenario | Target data during training? |
|---|---|
| Traditional | ✅ same dataset |
| Standard CDPR | ✅ unlabeled target included |
| **CDPR-UT (this paper)** | ❌ target is fully unseen |

The method has two generalization stages:

1. **Data-level** — Fourier-based augmentation mixes amplitude spectra across source datasets to simulate new visual styles (illumination, device response) while preserving palm-line semantics.

2. **Feature-level** — Four losses train domain-invariant, discriminative features:
   - `L_sup` (ArcFace) — supervised discrimination
   - `L_ada` (MK-MMD) — domain adaptation between source datasets
   - `L_con` — consistent loss: augmented features ≈ original features
   - `L_d-t` — dataset-aware triplet loss: intra-class similarity, inter-class variability

---

## Project Structure

```
pdfg/
├── train.py                  ← Entry point
├── trainer.py                ← Full training loop (PDFGTrainer)
├── requirements.txt
│
├── data/
│   ├── dataset.py            ← CASIAMultiSpectral, CASIASpectrum
│   └── fourier_augment.py    ← Fourier augmentation (Eqs. 2–7)
│
├── models/
│   └── feature_extractor.py  ← MultiDatasetExtractors (Fig. 3)
│
├── losses/
│   └── losses.py             ← ArcFace, MK-MMD, L_con, L_d-t, PDFGLoss
│
└── utils/
    └── metrics.py            ← Identification accuracy + EER
```

---

## Installation

```bash
pip install -r requirements.txt
```

GPU strongly recommended (paper used NVIDIA 2080Ti).

---

## Data Preparation

### CASIA Multi-Spectral Palmprint Database

The paper uses 6 spectra: **460, 630, 700, 850, 940, WHT** — 1,200 images each (200 hands).  
Since you already have ROI images, organize them as one of:

**Flat layout** (default, `subfolder_mode=False`):
```
CASIA-MS/
├── 460/
│   ├── 001_l_460_01.jpg    ← numeric prefix = subject ID
│   ├── 001_l_460_02.jpg
│   ├── 002_l_460_01.jpg
│   └── ...
├── 630/
│   └── ...
...
```

**Subfolder layout** (`--subfolder_mode`):
```
CASIA-MS/
├── 460/
│   ├── 001/
│   │   ├── 01.jpg
│   │   └── 02.jpg
│   ├── 002/
│   └── ...
├── 630/
...
```

---

## Usage

### Single experiment

```bash
# Sources: 460 + 630,  Target: 700  (matches a Table II row)
python train.py \
  --root /path/to/CASIA-MS \
  --sources 460 630 \
  --target 700

# 3-source experiment
python train.py \
  --root /path/to/CASIA-MS \
  --sources 460 630 700 \
  --target 850
```

### Run all Table II experiments

```bash
python train.py --root /path/to/CASIA-MS --run_all
```

### Subfolder layout

```bash
python train.py --root /path/to/CASIA-MS --sources 460 630 --target 700 --subfolder_mode
```

### Resume from checkpoint

```bash
python train.py --root /path/to/CASIA-MS --sources 460 630 --target 700 --resume
```

### All options

```
--root            Root directory of CASIA-MS ROI data         [required]
--sources         Source spectrum names                        e.g. 460 630 700
--target          Target spectrum name                         e.g. 850
--run_all         Run all Table II paper experiments

--subfolder_mode  Images are in per-class subfolders
--batch_size      Mini-batch size                              [default: 8]
--num_workers     DataLoader workers                           [default: 4]
--seed            Random seed                                  [default: 42]

--feature_dim     Output feature dimension                     [default: 128]
--alpha           L_con weight α                               [default: 0.1]
--beta            L_d-t weight β                               [default: 1.0]
--arcface_s       ArcFace scale s                              [default: 64.0]
--arcface_m       ArcFace angular margin m                     [default: 0.5]
--lam             Fourier augmentation λ                       [default: 0.8]

--epochs          Training epochs                              [default: 100]
--steps_per_epoch Steps per epoch (default: min loader length)
--lr              Learning rate                                [default: 1e-4]
--eval_every      Evaluate every N epochs                      [default: 5]
--resume          Resume from existing checkpoint
--cpu             Force CPU (for debugging)
--save_dir        Directory to save checkpoints & results      [default: runs/]
```

---

## Expected Results (from Table II)

| Sources | Target | Accuracy ↑ | EER ↓ |
|---|---|---|---|
| 460, 630 | 700 | 97.67% | 3.79% |
| 460, 630 | 850 | 93.33% | 3.72% |
| 460, 700, WHT | 630 | 99.33% | 1.69% |
| 460, WHT | 700 | 74.33% | 12.53% |
| **Average** | — | **92.82%** | **4.07%** |

---

## Architecture Details (Fig. 3)

```
Input [B, 1, 112, 112]
│
├── Shared Layers (all N datasets share these weights)
│   ├── Conv1  3×3×16, stride 4, Leaky ReLU
│   ├── MaxPool 2×2, stride 1
│   ├── Conv2  5×5×32, stride 2, Leaky ReLU
│   ├── MaxPool 2×2, stride 1
│   ├── Conv3  3×3×64, stride 1, Leaky ReLU
│   ├── Conv4  3×3×128, stride 1, Leaky ReLU
│   └── MaxPool 2×2, stride 1
│
└── Specific Layers × N  (one per source dataset)
    ├── FC1: → 1024, Leaky ReLU
    ├── FC2: → 512,  Leaky ReLU
    └── FC3: → 128   (L2 normalized output)
```

For inference on the target dataset, features from all N heads are **averaged** then L2-normalized.

---

## Loss Function (Eq. 11)

```
L = L_ArcFace + L_MK-MMD + α·L_con + β·L_d-t
```

| Loss | Purpose | Default weight |
|---|---|---|
| L_ArcFace | Discriminative features per dataset | 1.0 |
| L_MK-MMD | Reduce domain shift between source datasets | 1.0 |
| L_con | Augmented features ≈ original features | α = 0.1 |
| L_d-t | Triplet: intra-class sim, inter-class var | β = 1.0 |

---

## Citation

```bibtex
@article{shao2024pdfg,
  title   = {Learning to Generalize Unseen Dataset for Cross-Dataset Palmprint Recognition},
  author  = {Shao, Huikai and Zou, Yuchen and Liu, Chengcheng and Guo, Qiang and Zhong, Dexing},
  journal = {IEEE Transactions on Information Forensics and Security},
  volume  = {19},
  pages   = {3788--3799},
  year    = {2024},
  doi     = {10.1109/TIFS.2024.3371257}
}
```
