import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import math
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
# 2. Fourier Augmentation
# ----------------------------
def fourier_augment_batch(batch1, batch2, lam=0.8):
    """
    Mix amplitude spectra of batch1 (source) with batch2 (style donor).
    Augmented images keep the identity of batch1 with the style of batch2.
    """
    B, C, H, W = batch1.shape
    result = torch.zeros_like(batch1)
    b1, b2 = batch1.cpu().numpy(), batch2.cpu().numpy()
    for i in range(B):
        for c in range(C):
            F1 = np.fft.fft2(b1[i, c])
            F2 = np.fft.fft2(b2[i, c])
            A_mixed = (1 - lam) * np.abs(F1) + lam * np.abs(F2)
            F_new   = A_mixed * np.exp(1j * np.angle(F1))
            result[i, c] = torch.from_numpy(
                np.clip(np.real(np.fft.ifft2(F_new)), 0, 1).astype(np.float32)
            )
    return result.to(batch1.device)

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
            imgs  = list(imgs); rng.shuffle(imgs)
            sp    = max(1, int(len(imgs) * 0.75))
            chosen = imgs[:sp] if split == "train" else imgs[sp:]
            self.samples.extend((p, label) for p in chosen)

    def __len__(self):  return len(self.samples)
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
class ArcFaceHead(nn.Module):
    """ArcFace: Additive Angular Margin Loss (L_sup, Eq. 1)."""
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s, self.m   = s, m
        self.weight      = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m);  self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        cosine = F.linear(features, F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1)
        return F.cross_entropy((one_hot * phi + (1 - one_hot) * cosine) * self.s, labels)

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
    """L_con: augmented features averaged over all heads should match original (Eq. 9)."""
    loss = 0.0
    for head_feats in aug_feats_per_pair:           # head_feats: list of N tensors [B,d]
        avg = torch.stack(head_feats, dim=0).mean(0)
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
    B = anchor_labels.size(0)
    negatives = torch.zeros_like(aug_feats[:B])
    for i in range(B):
        mask = aug_labels != anchor_labels[i]
        pool = mask.nonzero(as_tuple=False).squeeze(1)
        negatives[i] = aug_feats[pool[torch.randint(len(pool), (1,))]] if len(pool) else aug_feats[random.randint(0, B-1)]
    return negatives

# ----------------------------
# 6. Data Loading
# ----------------------------
N = len(train_domains)

print("Building datasets...")
src_train = [CASIASpectrum(data_path, s, "train") for s in train_domains]
src_test  = [CASIASpectrum(data_path, s, "test")  for s in train_domains]
tgt_test  = CASIASpectrum(data_path, test_domains[0], "test")

num_classes_per_src = [ds.label_map.__len__() for ds in src_train]

train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)  for ds in src_train]
reg_loader    =  DataLoader(ConcatDataset(src_test), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
tgt_loader    =  DataLoader(tgt_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Infinite iterators so we can zip loaders of different lengths
class _Inf:
    def __init__(self, loader): self.loader = loader; self._it = iter(loader)
    def next(self):
        try: return next(self._it)
        except StopIteration: self._it = iter(self.loader); return next(self._it)

inf_loaders = [_Inf(ld) for ld in train_loaders]

# ----------------------------
# 7. Model & Optimizer Setup
# ----------------------------
model      = MultiDatasetExtractors(N, feature_dim).to(device)
arc_heads  = nn.ModuleList([ArcFaceHead(feature_dim, nc, arcface_s, arcface_m) for nc in num_classes_per_src]).to(device)
optimizer  = optim.Adam(list(model.parameters()) + list(arc_heads.parameters()), lr=lr)

steps_per_epoch = min(len(ld) for ld in train_loaders)

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

    # Identification accuracy
    sim  = torch.mm(tgt_f, reg_f.t())
    pred = reg_l[sim.argmax(dim=1)]
    acc  = (pred == tgt_l).float().mean().item() * 100

    # EER
    sim_mat = torch.mm(tgt_f, tgt_f.t()).numpy()
    n = len(tgt_l)
    gen, imp = [], []
    for i in range(n):
        for j in range(i+1, n):
            (gen if tgt_l[i] == tgt_l[j] else imp).append(sim_mat[i, j])
    gen, imp = np.array(gen), np.array(imp)
    thrs = np.linspace(min(gen.min(), imp.min()), max(gen.max(), imp.max()), 500)
    eer  = min(((abs((imp >= t).mean() - (gen < t).mean()), ((imp >= t).mean() + (gen < t).mean()) / 2) for t in thrs), key=lambda x: x[0])[1] * 100

    print(f"  Identification Acc : {acc:.2f}%")
    print(f"  EER                : {eer:.2f}%")
    return acc, eer

# ----------------------------
# 9. Training
# ----------------------------

# ── Phase 1: Supervised pre-training (L_sup only) ──────────────────────────
# Each head trained on its own dataset's original + Fourier-augmented images.
print(f"\n{'='*55}")
print(f"  Phase 1 — Supervised Pre-training  ({pretrain_epochs} epochs)")
print(f"{'='*55}")

for epoch in range(pretrain_epochs):
    model.train(); arc_heads.train()
    epoch_loss = 0.0

    for _ in tqdm(range(steps_per_epoch), desc=f"Pretrain {epoch+1}/{pretrain_epochs}"):
        batches = [(imgs.to(device), lbl.to(device)) for imgs, lbl in (il.next() for il in inf_loaders)]
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        for i, (src_imgs, src_lbl) in enumerate(batches):
            # Original images through head i
            loss += arc_heads[i](model.extract(src_imgs, i), src_lbl)
            # Fourier-augmented x^{Di->Dj} through head i (same identity labels)
            for j in range(N):
                if i == j: continue
                sty, _ = batches[j]
                if sty.size(0) != src_imgs.size(0):
                    sty = sty[torch.randint(sty.size(0), (src_imgs.size(0),))]
                aug_imgs = fourier_augment_batch(src_imgs, sty, lam)
                loss += arc_heads[i](model.extract(aug_imgs, i), src_lbl)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        epoch_loss += loss.item()

    print(f"  Epoch [{epoch+1}/{pretrain_epochs}] L_sup = {epoch_loss/steps_per_epoch:.4f}")

# ── Phase 2: Full PDFG training (all losses, Eq. 11) ───────────────────────
print(f"\n{'='*55}")
print(f"  Phase 2 — Full PDFG Training  ({epochs} epochs)")
print(f"{'='*55}")

best_eer = float("inf")

for epoch in range(epochs):
    model.train(); arc_heads.train()
    log = {"total": 0., "sup": 0., "ada": 0., "con": 0., "dt": 0.}

    for _ in tqdm(range(steps_per_epoch), desc=f"Train {epoch+1}/{epochs}"):
        batches = [(imgs.to(device), lbl.to(device)) for imgs, lbl in (il.next() for il in inf_loaders)]

        # Build all Fourier-augmented pairs (Di -> Dj)
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

        # L_sup: original + augmented images through their own head
        orig_feats = []
        for i, (src_imgs, src_lbl) in enumerate(batches):
            feat = model.extract(src_imgs, i)
            l_sup = arc_heads[i](feat, src_lbl)
            orig_feats.append(feat)
            loss += l_sup; log["sup"] += l_sup.item()

            for j in range(N):
                if i == j: continue
                aug_imgs, aug_lbl = aug[(i, j)]
                l_sup_aug = arc_heads[i](model.extract(aug_imgs, i), aug_lbl) / (N - 1)
                loss += l_sup_aug; log["sup"] += l_sup_aug.item()

        # L_con + L_d-t: per source dataset
        for i in range(N):
            aug_head_feats, aug_labels_list = [], []
            for j in range(N):
                if i == j: continue
                aug_imgs, aug_lbl = aug[(i, j)]
                aug_head_feats.append(model.extract_all(aug_imgs))   # list of N [B,d]
                aug_labels_list.append(aug_lbl)

            l_con = consistent_loss(orig_feats[i], aug_head_feats)
            loss += l_con; log["con"] += l_con.item()

            aug_avg = torch.stack([torch.stack(hf, 0).mean(0) for hf in aug_head_feats], 0).mean(0)
            neg     = sample_negative(aug_avg, aug_labels_list[0], batches[i][1])
            l_dt    = triplet_loss(orig_feats[i], torch.roll(orig_feats[i], 1, 0), neg, triplet_margin)
            loss += l_dt; log["dt"] += l_dt.item()

        # L_ada: MK-MMD between every source pair
        ada_count = 0
        for i in range(N):
            for j in range(i+1, N):
                l_ada = mkmmd_loss(orig_feats[i], orig_feats[j])
                loss += l_ada; log["ada"] += l_ada.item(); ada_count += 1
        if ada_count: log["ada"] /= ada_count

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        log["total"] += loss.item()

    for k in log: log[k] /= steps_per_epoch
    print(f"  Epoch [{epoch+1}/{epochs}] total={log['total']:.4f} | sup={log['sup']:.4f} | ada={log['ada']:.4f} | con={log['con']:.4f} | dt={log['dt']:.4f}")

    if (epoch + 1) % eval_every == 0:
        print(f"\n  -- Evaluation at epoch {epoch+1} --")
        acc, eer = evaluate()
        if eer < best_eer:
            best_eer = eer
            torch.save({"model": model.state_dict(), "heads": arc_heads.state_dict()}, "best_model.pth")
            print(f"  New best EER: {eer:.2f}% -> saved best_model.pth")
        print()

print(f"\nDone. Best EER: {best_eer:.2f}%")
