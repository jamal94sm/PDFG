# PDFG — Palmprint Data and Feature Generation

PyTorch implementation of:

> **"Learning to Generalize Unseen Dataset for Cross-Dataset Palmprint Recognition"**  
> Shao et al., IEEE Transactions on Information Forensics and Security, Vol. 19, 2024  
> DOI: 10.1109/TIFS.2024.3371257

---

## Overview

PDFG tackles **CDPR-UT** (Cross-Dataset Palmprint Recognition with **Unseen** Target dataset) — the realistic scenario where the target domain is completely unavailable during training.

| Scenario | Target data during training? |
|---|---|
| Traditional | ✅ same dataset |
| Standard CDPR | ✅ unlabeled target included |
| **CDPR-UT (this paper)** | ❌ target is fully unseen |

Two generalization stages are combined:

1. **Data-level** — Fourier augmentation mixes amplitude spectra across source datasets, generating new styles while preserving palm-line semantics (phase spectrum).
2. **Feature-level** — Four losses train domain-invariant, discriminative features:
   - `L_sup` — ArcFace supervised discrimination (one head per source dataset)
   - `L_ada` — MK-MMD domain adaptation between source datasets
   - `L_con` — consistent loss: augmented features ≈ original features
   - `L_d-t` — dataset-aware triplet loss: intra-class similarity, inter-class variability

---

## Project Structure

```
pdfg/
├── train.py                  ← Entry point (CLI)
├── trainer.py                ← Two-phase PDFGTrainer
├── requirements.txt
│
├── data/
│   ├── dataset.py            ← CASIASpectrum, CASIAMultiSpectral
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
pip install torch torchvision pillow numpy
```

GPU strongly recommended (paper used NVIDIA 2080Ti).

---

## Data Preparation

### CASIA Multi-Spectral Palmprint Database

6 spectra: **460, 630, 700, 850, 940, WHT** — 1,200 images each (200 hands × 6 images).

The dataset loader expects a **single flat folder** containing all `.jpg` files, named with this convention:

```
{subject_id}_{hand}_{spectrum}_{iteration}.jpg
```

Example:
```
/home/pai-ng/Jamal/CASIA-MS-ROI/
├── 001_l_460_01.jpg
├── 001_l_460_02.jpg
├── 001_r_460_01.jpg
├── 002_l_460_01.jpg
├── 001_l_700_01.jpg
├── 001_l_630_01.jpg
└── ...
```

**Identity label** = `subject_id + "_" + hand` (e.g. `"001_l"` and `"001_r"` are two separate identities).  
The loader automatically filters by spectrum and groups by identity — no manual pre-sorting needed.

**Train/test split**: 3/4 of each identity's images for training, 1/4 for testing (paper convention).

---

## Training Procedure

Training runs in two phases, exactly as described in the paper (Section III-B):

### Phase 1 — Supervised Pre-training (`L_sup` only)

> *"The feature extractors are firstly trained by L_sup using the labeled source images and augmented images."*

- N feature extractors (one per source spectrum), sharing the CNN backbone
- Each head trained with ArcFace on its own dataset's identity labels
- Input = **original** source images **+ Fourier-augmented** images `x^{Di→Dj}`
  - Augmented images carry the same identity label as their source → same head, same class space
- Shared CNN receives gradients from all N heads simultaneously

### Phase 2 — Full PDFG Training (Eq. 11)

```
L = L_ArcFace + L_MK-MMD + α·L_con + β·L_d-t
```

Continues from pre-trained weights, adding cross-dataset losses:

| Loss | Purpose | Default weight |
|---|---|---|
| `L_ArcFace` | Discriminative features per dataset | 1.0 |
| `L_MK-MMD` | Reduce domain shift between source datasets | 1.0 |
| `L_con` | Augmented features ≈ original features (Eq. 9) | α = 0.1 |
| `L_d-t` | Triplet: intra-class sim, inter-class var (Eq. 10) | β = 1.0 |

---

## Usage

### Default run (matches your existing train/test domain setup)

```bash
python train.py
# equivalent to:
python train.py --sources 460 700 630 --target 940
```

### Custom experiment

```bash
python train.py --sources 460 630 --target 700
python train.py --sources 460 630 700 --target 850
```

### Run all Table II experiments

```bash
python train.py --run_all
```

### Resume from checkpoint

```bash
python train.py --sources 460 700 630 --target 940 --resume
```

### All options

```
Data
  --root             Flat folder with all .jpg ROI files
                     [default: /home/pai-ng/Jamal/CASIA-MS-ROI]
  --sources          Source spectrum names    [default: 460 700 630]
  --target           Target spectrum name     [default: 940]
  --run_all          Run all Table II paper experiments
  --batch_size       Mini-batch size          [default: 8]
  --num_workers      DataLoader workers       [default: 2]
  --seed             Random seed              [default: 42]

Model
  --feature_dim      Output feature dimension [default: 128]

Loss hyperparameters (paper defaults)
  --alpha            L_con weight α           [default: 0.1]
  --beta             L_d-t weight β           [default: 1.0]
  --arcface_s        ArcFace scale s          [default: 64.0]
  --arcface_m        ArcFace angular margin m [default: 0.5]
  --lam              Fourier augmentation λ   [default: 0.8]

Training
  --pretrain_epochs  Phase 1 epochs           [default: 30]
  --epochs           Phase 2 epochs           [default: 100]
  --steps_per_epoch  Steps per epoch          [default: min loader length]
  --lr               Learning rate            [default: 1e-4]
  --eval_every       Evaluate every N epochs  [default: 5]
  --resume           Resume from checkpoint
  --cpu              Force CPU (for debugging)
  --save_dir         Checkpoint directory     [default: runs/]
```

---

## Architecture (Fig. 3)

```
Input [B, 3, 224, 224]   (RGB, 224×224)
│
├── Shared Layers  (weights shared across all N source datasets)
│   ├── Conv1  3×3, 16 ch, stride 4, Leaky ReLU
│   ├── MaxPool 2×2, stride 1
│   ├── Conv2  5×5, 32 ch, stride 2, Leaky ReLU
│   ├── MaxPool 2×2, stride 1
│   ├── Conv3  3×3, 64 ch, stride 1, Leaky ReLU
│   ├── Conv4  3×3, 128 ch, stride 1, Leaky ReLU
│   └── MaxPool 2×2, stride 1
│
└── Specific Layers × N  (one FC head per source dataset)
    ├── FC1  → 1024, Leaky ReLU
    ├── FC2  → 512,  Leaky ReLU
    └── FC3  → 128   (L2-normalized output)
```

At inference on the target dataset, features from **all N heads are averaged** then L2-normalized — the paper's approach for producing the final feature.

---

## Expected Results (Table II, CASIA Multi-Spectral)

| Sources | Target | Accuracy ↑ | EER ↓ |
|---|---|---|---|
| 460, 630 | 700 | 97.67% | 3.79% |
| 460, 630 | 850 | 93.33% | 3.72% |
| 460, 700, WHT | 630 | 99.33% | 1.69% |
| 460, WHT | 700 | 74.33% | 12.53% |
| **Average** | — | **92.82%** | **4.07%** |

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
