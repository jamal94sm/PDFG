import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_metric_learning import losses as pml_losses
from tqdm import tqdm

# ----------------------------
# 1. Configuration
# ----------------------------
data_path     = "/home/pai-ng/Jamal/CASIA-MS-ROI"
train_domains = ["460", "700", "630"]
test_domains  = ["940"]

batch_size      = 8
lr              = 1e-4
epochs          = 100
pretrain_epochs = 30
arcface_s       = 64.0
arcface_m       = 0.5
triplet_margin  = 0.4
alpha           = 0.1   # L_con weight
beta            = 1.0   # L_d-t weight
lam             = 0.8   # Fourier augmentation λ
feature_dim     = 128
eval_every      = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Fourier Augmentation (EDITED for GPU speed)
# ----------------------------
def fourier_augment_batch(batch1, batch2, lam=0.8):
    """
    Generate x^{D1->D2}: same identity as batch1, new style from batch2.
    Fully vectorized in PyTorch for GPU acceleration.
    """
    # Perform 2D FFT on the last two dimensions (H, W)
    fft1 = torch.fft.fft2(batch1, dim=(-2, -1))
    fft2 = torch.fft.fft2(batch2, dim=(-2, -1))
    
    # Extract amplitude and phase
    amp1, phase1 = torch.abs(fft1), torch.angle(fft1)
    amp2 = torch.abs(fft2)
    
    # Interpolate amplitude (style) and keep original phase (identity)
    amp_mixed = (1 - lam) * amp1 + lam * amp2
    
    # Reconstruct the complex tensor
    fft_new = amp_mixed * torch.exp(1j * phase1)
    
    # Inverse FFT to get back to the spatial domain
    result = torch.real(torch.fft.ifft2(fft_new, dim=(-2, -1)))
    
    # Clamp to valid image range
    return torch.clamp(result, 0.0, 1.0)

# ----------------------------
# 3. Dataset
# ----------------------------
class CASIASpectrum(Dataset):
    """
    Single-spectrum loader from a flat folder.
    Filename: {subject_id}_{hand}_{spectrum}_{iteration}.jpg
    Identity label = subject_id + "_" + hand  (e.g. "001_l")
    Split: 3/4 train, 1/4 test per identity.
    """
    def __init__(self, data_path, spectrum, split="train", seed=42):
        self.to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        class_to_imgs  = {}

        for fname in sorted(os.listdir(data_path)):
            if not fname.lower().endswith(".jpg"):
                continue
            parts = fname[:-4].split("_")
            if len(parts) != 4:
                continue
            subject_id, hand, spec, _ = parts
            if spec != spectrum:
                continue
            hand_id = f"{subject_id}_{hand}"
            class_to_imgs.setdefault(hand_id, []).append(os.path.join(data_path, fname))

        self.label_map = {h: i for i, h in enumerate(sorted(class_to_imgs))}
        self.samples   = []
        for hand_id, imgs in class_to_imgs.items():
            label = self.label_map[hand_id]
            rng   = random.Random(seed + label)
            imgs  = list(imgs)
            rng.shuffle(imgs)
            sp     = max(1, int(len(imgs) * 0.75))
            chosen = imgs[:sp] if split == "train" else imgs[sp:]
            self.samples.extend((p, label) for p in chosen)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.to_tensor(Image.open(path).convert("RGB")), label

# ----------------------------
# 4. Model
# ----------------------------
class SharedLayers(nn.Module):
    """Shared CNN backbone — weights shared across all N feature extractor heads."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = self.pool3(self.act(self.conv4(x)))
        return x

class MultiDatasetExtractors(nn.Module):
    """
    N feature extractors sharing the CNN backbone, each with its own FC head.
    At inference, features from all N heads are averaged (paper Section III).
    """
    def __init__(self, n_datasets, feature_dim=128):
        super().__init__()
        self.n_datasets = n_datasets
        self.shared     = SharedLayers()
        with torch.no_grad():
            flat_dim = self.shared(torch.zeros(1, 3, 224, 224)).view(1, -1).shape[1]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 1024), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 512),      nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, feature_dim),
            ) for _ in range(n_datasets)
        ])

    def extract(self, x, idx):
        f = self.heads[idx](self.shared(x).view(x.size(0), -1))
        return F.normalize(f, p=2, dim=1)

    def extract_all(self, x):
        shared = self.shared(x).view(x.size(0), -1)
        return [F.normalize(h(shared), p=2, dim=1) for h in self.heads]

# ----------------------------
# 5. Losses
# ----------------------------
# We use pytorch_metric_learning ArcFaceLoss instead of a custom one.
# The math is identical (additive angular margin, Eq. 1 in paper).
# Key advantage: exposes get_logits() so we can compute train accuracy
# from already-computed features with no extra forward pass.

def mkmmd_loss(src, tgt, kernels=(1, 5, 10, 20, 50, 100)):
    """MK-MMD domain adaptation loss (L_ada, Eq. 8)."""
    loss = 0.0
    for bw in kernels:
        k_ss = torch.exp(-torch.cdist(src, src).pow(2) / bw).mean()
        k_st = torch.exp(-torch.cdist(src, tgt).pow(2) / bw).mean()
        k_tt = torch.exp(-torch.cdist(tgt, tgt).pow(2) / bw).mean()
        loss += k_ss - 2 * k_st + k_tt
    return loss / len(kernels)

def consistent_loss(orig_feat, aug_feats_per_pair):
    """
    L_con (Eq. 9): for each augmented pair x^{Dj->Dn}, pass through ALL N heads,
    average the N features, then penalise distance from the original feature f(x^Dj)^j.
    aug_feats_per_pair: list of (N-1) elements, each a list of N tensors [B, d].
    """
    loss = 0.0
    for head_feats in aug_feats_per_pair:
        avg   = torch.stack(head_feats, dim=0).mean(0)
        loss += F.mse_loss(orig_feat, avg)
    return loss / max(len(aug_feats_per_pair), 1)

def triplet_loss(anchor, positive, negative, margin=0.4):
    """L_d-t: dataset-aware triplet loss (Eq. 10)."""
    return F.relu(
        F.pairwise_distance(anchor, positive) -
        F.pairwise_distance(anchor, negative) + margin
    ).mean()

def sample_negative(aug_feats, aug_labels, anchor_labels):
    """Pick a different-class augmented feature for each anchor."""
    B         = anchor_labels.size(0)
    negatives = torch.zeros_like(aug_feats[:B])
    for i in range(B):
        pool = (aug_labels != anchor_labels[i]).nonzero(as_tuple=False).squeeze(1)
        negatives[i] = aug_feats[pool[torch.randint(len(pool), (1,))]] if len(pool) else aug_feats[random.randint(0, B - 1)]
    return negatives

# ----------------------------
# 6. Data Loading
# ----------------------------
N = len(train_domains)

print("Building datasets...")
src_train = [CASIASpectrum(data_path, s, "train") for s in train_domains]
src_test  = [CASIASpectrum(data_path, s, "test")  for s in train_domains]
tgt_test  = CASIASpectrum(data_path, test_domains[0], "test")

num_classes_per_src = [len(ds.label_map) for ds in src_train]

train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True) for ds in src_train]
reg_loader    =  DataLoader(ConcatDataset(src_test), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
tgt_loader    =  DataLoader(tgt_test,                batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

class _Inf:
    """Wraps a DataLoader so it never raises StopIteration."""
    def __init__(self, loader): self.loader = loader; self._it = iter(loader)
    def next(self):
        try: return next(self._it)
        except StopIteration: self._it = iter(self.loader); return next(self._it)

inf_loaders = [_Inf(ld) for ld in train_loaders]

# ----------------------------
# 7. Model & Optimizer Setup
# ----------------------------
model     = MultiDatasetExtractors(N, feature_dim).to(device)
arc_heads = nn.ModuleList([
    pml_losses.ArcFaceLoss(num_classes=nc, embedding_size=feature_dim, margin=arcface_m, scale=arcface_s).to(device)
    for nc in num_classes_per_src
])

# all_params used for optimiser AND grad clipping — ArcFace weights must be included
all_params = list(model.parameters()) + list(arc_heads.parameters())
optimizer  = optim.Adam(all_params, lr=lr)

steps_per_epoch = min(len(ld) for ld in train_loaders)

print(f"  Classes per source : {num_classes_per_src}")
print(f"  Steps per epoch    : {steps_per_epoch}")
print(f"  Device             : {device}\n")

# ----------------------------
# 8. Evaluation
# ----------------------------
@torch.no_grad()
def evaluate():
    model.eval()
    def get_feats(loader):
        all_f, all_l = [], []
        for imgs, labels in loader:
            imgs = imgs.to(device)
            fs   = torch.stack(model.extract_all(imgs), dim=0).mean(0)
            all_f.append(F.normalize(fs, p=2, dim=1).cpu())
            all_l.append(labels)
        return torch.cat(all_f), torch.cat(all_l)

    reg_f, reg_l = get_feats(reg_loader)
    tgt_f, tgt_l = get_feats(tgt_loader)

    sim  = torch.mm(tgt_f, reg_f.t())
    pred = reg_l[sim.argmax(dim=1)]
    acc  = (pred == tgt_l).float().mean().item() * 100

    sim_mat = torch.mm(tgt_f, tgt_f.t()).numpy()
    gen, imp = [], []
    for i in range(len(tgt_l)):
        for j in range(i + 1, len(tgt_l)):
            (gen if tgt_l[i] == tgt_l[j] else imp).append(sim_mat[i, j])
    gen, imp = np.array(gen), np.array(imp)
    thrs = np.linspace(min(gen.min(), imp.min()), max(gen.max(), imp.max()), 500)
    eer  = min(
        ((abs((imp >= t).mean() - (gen < t).mean()), ((imp >= t).mean() + (gen < t).mean()) / 2) for t in thrs),
        key=lambda x: x[0]
    )[1] * 100

    print(f"  -> Test  | Acc: {acc:.2f}%  EER: {eer:.2f}%")
    return acc, eer

# ----------------------------
# 9. Training
# ----------------------------

# ── Phase 1: Supervised pre-training (L_sup only) ──────────────────────────
#
# Paper (Section III-B): "The feature extractors are firstly trained by L_sup
# using the labeled source images and augmented images."
#
# Augmented images x^{Di->Dj} (style of Dj, identity of Di) are fed through
# head i with Di's labels. This is NOT a paradox — it is intentional:
#   - Phase spectrum (identity info) is preserved from Di
#   - Amplitude spectrum (style/illumination) is mixed toward Dj
# Training head i on these forces it to be STYLE-INVARIANT: it must
# recognise the same identity whether it looks like spectrum 460 or 700.
# This is exactly the generalisation the paper is after.
#
print(f"{'='*55}")
print(f"  Phase 1 — Supervised Pre-training  ({pretrain_epochs} epochs)")
print(f"  (L_sup on original + Fourier-augmented images per head)")
print(f"{'='*55}")

for epoch in range(pretrain_epochs):
    model.train()
    for h in arc_heads: h.train()
    epoch_loss  = 0.0
    epoch_corr  = 0
    epoch_total = 0

    for _ in tqdm(range(steps_per_epoch), desc=f"Pretrain {epoch+1}/{pretrain_epochs}", leave=False):
        batches = [(imgs.to(device), lbl.to(device)) for imgs, lbl in (il.next() for il in inf_loaders)]
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        for i, (src_imgs, src_lbl) in enumerate(batches):
            # Original source images → head i with Di's labels
            feat  = model.extract(src_imgs, i)
            loss += arc_heads[i](feat, src_lbl)

            # Train accuracy from original images (no extra forward pass)
            with torch.no_grad():
                preds = arc_heads[i].get_logits(feat).argmax(dim=1)
                epoch_corr  += (preds == src_lbl).sum().item()
                epoch_total += src_lbl.size(0)

            # Augmented x^{Di->Dj} → head i with SAME Di labels
            # Head i must learn to be invariant to style changes
            for j in range(N):
                if i == j: continue
                sty, _ = batches[j]
                if sty.size(0) != src_imgs.size(0):
                    sty = sty[torch.randint(sty.size(0), (src_imgs.size(0),))]
                aug_feat = model.extract(fourier_augment_batch(src_imgs, sty, lam), i)
                loss    += arc_heads[i](aug_feat, src_lbl)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)   # clip ALL params incl. ArcFace weights
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / steps_per_epoch
    avg_acc  = 100.0 * epoch_corr / epoch_total if epoch_total > 0 else 0.0
    print(f"  Epoch [{epoch+1}/{pretrain_epochs}] Loss: {avg_loss:.4f}  Train Acc: {avg_acc:.2f}%")

# ── Phase 2: Full PDFG training (all losses, Eq. 11) ───────────────────────
#
# Paper (Section III-D, Eq. 9): for L_con, x^{Di->Dj} is fed into ALL N heads.
# This is different from L_sup where it only goes through its source head.
# The two usages of augmented images serve different purposes:
#   L_sup  : augmented → own head    → style invariance per extractor
#   L_con  : augmented → all heads   → cross-extractor feature consistency
#
print(f"\n{'='*55}")
print(f"  Phase 2 — Full PDFG Training  ({epochs} epochs, Eq. 11)")
print(f"  L = L_sup + L_ada + α·L_con + β·L_d-t")
print(f"{'='*55}")

best_eer = float("inf")

for epoch in range(epochs):
    model.train()
    for h in arc_heads: h.train()
    log         = {"total": 0., "sup": 0., "ada": 0., "con": 0., "dt": 0.}
    epoch_corr  = 0
    epoch_total = 0

    for _ in tqdm(range(steps_per_epoch), desc=f"Train {epoch+1}/{epochs}", leave=False):
        batches = [(imgs.to(device), lbl.to(device)) for imgs, lbl in (il.next() for il in inf_loaders)]

        # Build all Fourier-augmented pairs x^{Di->Dj} for i != j
        aug = {}
        for i in range(N):
            src_imgs, src_lbl = batches[i]
            for j in range(N):
                if i == j: continue
                sty, _ = batches[j]
                if sty.size(0) != src_imgs.size(0):
                    sty = sty[torch.randint(sty.size(0), (src_imgs.size(0),))]
                aug[(i, j)] = (fourier_augment_batch(src_imgs, sty, lam), src_lbl)

        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        # L_sup: original images → own head; augmented x^{Di->Dj} → head i (same labels)
        orig_feats = []
        for i, (src_imgs, src_lbl) in enumerate(batches):
            feat  = model.extract(src_imgs, i)
            l_sup = arc_heads[i](feat, src_lbl)
            orig_feats.append(feat)
            loss += l_sup
            log["sup"] += l_sup.item()

            with torch.no_grad():
                preds = arc_heads[i].get_logits(feat).argmax(dim=1)
                epoch_corr  += (preds == src_lbl).sum().item()
                epoch_total += src_lbl.size(0)

            for j in range(N):
                if i == j: continue
                aug_imgs, aug_lbl = aug[(i, j)]
                l_sup_aug  = arc_heads[i](model.extract(aug_imgs, i), aug_lbl)
                loss      += l_sup_aug
                log["sup"] += l_sup_aug.item()

        # L_con + L_d-t: augmented x^{Di->Dj} → ALL N heads (Eq. 9)
        for i in range(N):
            aug_head_feats, aug_labels_list = [], []
            for j in range(N):
                if i == j: continue
                aug_imgs, aug_lbl = aug[(i, j)]
                aug_head_feats.append(model.extract_all(aug_imgs))  # list of N [B,d]
                aug_labels_list.append(aug_lbl)

            l_con  = alpha * consistent_loss(orig_feats[i], aug_head_feats)
            loss  += l_con
            log["con"] += l_con.item()

            aug_avg = torch.stack([torch.stack(hf, 0).mean(0) for hf in aug_head_feats], 0).mean(0)
            neg     = sample_negative(aug_avg, aug_labels_list[0], batches[i][1])
            l_dt    = beta * triplet_loss(orig_feats[i], torch.roll(orig_feats[i], 1, 0), neg, triplet_margin)
            loss   += l_dt
            log["dt"] += l_dt.item()

        # L_ada: MK-MMD between every pair of source datasets
        for i in range(N):
            for j in range(i + 1, N):
                l_ada = mkmmd_loss(orig_feats[i], orig_feats[j])
                loss += l_ada
                log["ada"] += l_ada.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        log["total"] += loss.item()

    for k in log: log[k] /= steps_per_epoch
    avg_acc = 100.0 * epoch_corr / epoch_total if epoch_total > 0 else 0.0
    print(
        f"  Epoch [{epoch+1}/{epochs}] "
        f"Loss: {log['total']:.4f}  Train Acc: {avg_acc:.2f}%  |  "
        f"sup={log['sup']:.4f}  ada={log['ada']:.4f}  con={log['con']:.4f}  dt={log['dt']:.4f}"
    )

    if (epoch + 1) % eval_every == 0:
        acc, eer = evaluate()
        if eer < best_eer:
            best_eer = eer
            torch.save({"model": model.state_dict(), "heads": [h.state_dict() for h in arc_heads]}, "best_model.pth")
            print(f"  New best EER: {eer:.2f}% -> saved best_model.pth")
        print()

print(f"\nDone. Best EER: {best_eer:.2f}%")
