import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses as pml_losses
from tqdm import tqdm

# ----------------------------
# 1. Configuration
# ----------------------------
data_path     = "/home/pai-ng/Jamal/CASIA-MS-ROI"
train_domains = ["460", "WHT"]
test_domains  = ["700"]

batch_size      = 8
lr              = 1e-4
epochs          = 200
pretrain_epochs = 20
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
# 2. Fourier Augmentation (GPU-vectorised)
# ----------------------------
def fourier_augment_batch(batch1, batch2, lam=0.8):
    """
    Generate x^{D1->D2}: same identity (phase) as batch1, new style
    (amplitude) from batch2. Fully vectorised on GPU via torch.fft.
    """
    fft1 = torch.fft.fft2(batch1, dim=(-2, -1))
    fft2 = torch.fft.fft2(batch2, dim=(-2, -1))

    amp1, phase1 = torch.abs(fft1), torch.angle(fft1)
    amp2         = torch.abs(fft2)

    amp_mixed = (1 - lam) * amp1 + lam * amp2
    fft_new   = amp_mixed * torch.exp(1j * phase1)

    result = torch.real(torch.fft.ifft2(fft_new, dim=(-2, -1)))
    return torch.clamp(result, 0.0, 1.0)

# ----------------------------
# 3. Dataset
# ----------------------------
class CASIASpectrum(Dataset):
    """
    Loads ALL images belonging to the given list of spectra (domains).
    Filename: {subject_id}_{hand}_{spectrum}_{iteration}.jpg
    Identity label = subject_id + "_" + hand  (e.g. "001_l").

    A shared_label_map is passed in so that train and test datasets use
    the same integer label space. Train and test share the same identities,
    so the map is built from train domains and passed to all datasets.
    This ensures ArcFace head argmax predictions directly match test labels.
    """
    def __init__(self, data_path, spectra, shared_label_map):
        self.to_tensor = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        spectra_set   = set(spectra)
        class_to_imgs = {}

        for fname in sorted(os.listdir(data_path)):
            if not fname.lower().endswith(".jpg"):
                continue
            parts = fname[:-4].split("_")
            if len(parts) != 4:
                continue
            subject_id, hand, spec, _ = parts
            if spec not in spectra_set:
                continue
            hand_id = f"{subject_id}_{hand}"
            class_to_imgs.setdefault(hand_id, []).append(
                os.path.join(data_path, fname)
            )

        self.label_map = shared_label_map
        self.samples   = []
        for hand_id, imgs in class_to_imgs.items():
            if hand_id not in self.label_map:
                continue
            label = self.label_map[hand_id]
            self.samples.extend((p, label) for p in imgs)

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
            flat_dim = self.shared(torch.zeros(1, 3, 112, 112)).view(1, -1).shape[1]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 1024), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 512),      nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, feature_dim),
            ) for _ in range(n_datasets)
        ])

    def extract(self, x, idx):
        """Extract & L2-normalise features using head `idx`."""
        f = self.heads[idx](self.shared(x).view(x.size(0), -1))
        return F.normalize(f, p=2, dim=1)

    def extract_all(self, x):
        """Extract features from all N heads, each L2-normalised."""
        shared = self.shared(x).view(x.size(0), -1)
        return [F.normalize(h(shared), p=2, dim=1) for h in self.heads]

# ----------------------------
# 5. Losses
# ----------------------------
def mkmmd_loss(f1, f2, kernels=(1, 5, 10, 20, 50, 100)):
    """
    MK-MMD domain adaptation loss (L_ada, Eq. 8) on EXTRACTED FEATURES.
    Squared distances via ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y> with clamp.
    """
    def sq_dists(a, b):
        aa = (a * a).sum(dim=1, keepdim=True)
        bb = (b * b).sum(dim=1, keepdim=True)
        ab = torch.mm(a, b.t())
        return (aa + bb.t() - 2 * ab).clamp(min=0)

    d_ss = sq_dists(f1, f1)
    d_st = sq_dists(f1, f2)
    d_tt = sq_dists(f2, f2)

    loss = 0.0
    for bw in kernels:
        k_ss = torch.exp(-d_ss / bw).mean()
        k_st = torch.exp(-d_st / bw).mean()
        k_tt = torch.exp(-d_tt / bw).mean()
        loss += k_ss - 2 * k_st + k_tt
    return loss / len(kernels)

def consistent_loss(orig_feat, aug_feats_per_pair):
    """
    L_con (Eq. 9):
        L_con = SUM_{n!=j} || f(x^Dj)^j - (1/N) SUM_l f(x^{Dj->Dn})^l ||^2
    Outer: SUM (not mean) over N-1 pairs.
    Inner: squared L2 norm, averaged over batch.
    """
    loss = torch.tensor(0.0, device=orig_feat.device)
    for head_feats in aug_feats_per_pair:
        avg   = torch.stack(head_feats, dim=0).mean(0)
        sq_l2 = ((orig_feat - avg) ** 2).sum(dim=1)
        loss += sq_l2.mean()
    return loss

def triplet_loss_fn(anchor, positive, negative, margin=0.4):
    """L_d-t: dataset-aware triplet loss (Eq. 10)."""
    return F.relu(
        F.pairwise_distance(anchor, positive) -
        F.pairwise_distance(anchor, negative) + margin
    ).mean()

def sample_triplet_pairs(aug_avg, aug_labels, anchor_labels):
    """
    Positive: aug_avg[k] — same identity as anchor[k] by Fourier phase preservation.
    Negative: aug_avg[m] where aug_labels[m] != anchor_labels[k].
    """
    B         = anchor_labels.size(0)
    positives = aug_avg.clone()
    negatives = torch.zeros_like(aug_avg)
    for i in range(B):
        pool = (aug_labels != anchor_labels[i]).nonzero(as_tuple=False).squeeze(1)
        if len(pool) > 0:
            negatives[i] = aug_avg[pool[torch.randint(len(pool), (1,))]]
        else:
            negatives[i] = aug_avg[random.randint(0, B - 1)]
    return positives, negatives

# ----------------------------
# 6. EER helper
# ----------------------------
def compute_eer(scores_gen, scores_imp):
    """
    Compute EER from arrays of genuine and impostor similarity scores.
    Returns EER as a percentage.
    """
    if len(scores_gen) == 0 or len(scores_imp) == 0:
        return float("nan")
    gen  = np.array(scores_gen)
    imp  = np.array(scores_imp)
    thrs = np.linspace(min(gen.min(), imp.min()), max(gen.max(), imp.max()), 500)
    eer  = min(
        ((abs((imp >= t).mean() - (gen < t).mean()),
          ((imp >= t).mean() + (gen < t).mean()) / 2)
         for t in thrs),
        key=lambda x: x[0],
    )[1] * 100
    return eer

# ----------------------------
# 7. Data Loading
# ----------------------------
N = len(train_domains)

print("Building datasets...")

# ── Shared label map built from TRAIN domains ────────────────────────────────
# Train and test sets share the same identities, so scanning train domains
# is sufficient to build the full label map.
# All datasets (train + test) receive this shared map so that integer label k
# refers to the same hand_id everywhere — essential for ArcFace head accuracy:
# source head argmax predictions directly correspond to test label integers.
all_hand_ids = set()
for fname in sorted(os.listdir(data_path)):
    if not fname.lower().endswith(".jpg"):
        continue
    parts = fname[:-4].split("_")
    if len(parts) != 4:
        continue
    subject_id, hand, spec, _ = parts
    if spec in set(train_domains):       # only scan train domains
        all_hand_ids.add(f"{subject_id}_{hand}")
shared_label_map  = {h: i for i, h in enumerate(sorted(all_hand_ids))}
num_total_classes = len(shared_label_map)
print(f"  Shared identity space: {num_total_classes} identities (same in train & test)")

src_train = [CASIASpectrum(data_path, [s], shared_label_map) for s in train_domains]
tgt_test  = CASIASpectrum(data_path, test_domains, shared_label_map)

train_loaders = [
    DataLoader(ds, batch_size=batch_size, shuffle=True,
               num_workers=0, pin_memory=False, drop_last=True)
    for ds in src_train
]
test_loader = DataLoader(tgt_test, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)

# ── Split test set: gallery (1st image per identity) + query (rest) ──────────
_tf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])

class _ListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label

gallery_samples, query_samples = [], []
_seen = set()
for path, label in tgt_test.samples:
    if label not in _seen:
        _seen.add(label)
        gallery_samples.append((path, label))
    else:
        query_samples.append((path, label))

gallery_loader = DataLoader(_ListDataset(gallery_samples, _tf),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
query_loader   = DataLoader(_ListDataset(query_samples, _tf),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

steps_per_epoch = min(len(ld) for ld in train_loaders)

class _Inf:
    def __init__(self, loader):
        self.loader = loader
        self._it    = iter(loader)
    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)

inf_loaders = [_Inf(ld) for ld in train_loaders]

print(f"  Train domains     : {train_domains}  (N={N} heads)")
print(f"  Test  domain      : {test_domains[0]}")
print(f"  Train samples     : {[len(ds) for ds in src_train]}")
print(f"  Test  samples     : {len(tgt_test)}  "
      f"(gallery={len(gallery_samples)}, query={len(query_samples)})")
print(f"  Steps per epoch   : {steps_per_epoch}")
print(f"  Device            : {device}\n")

# ----------------------------
# 8. Model & Optimizer Setup
# ----------------------------
model = MultiDatasetExtractors(N, feature_dim).to(device)

# ArcFace heads use num_total_classes (shared space) so argmax predictions
# are directly comparable to test set label integers.
arc_heads = nn.ModuleList([
    pml_losses.ArcFaceLoss(
        num_classes=num_total_classes, embedding_size=feature_dim,
        margin=arcface_m, scale=arcface_s,
    ).to(device)
    for _ in range(N)
])

all_params = list(model.parameters()) + list(arc_heads.parameters())
optimizer  = optim.Adam(all_params, lr=lr)

# ----------------------------
# 9. Evaluation
# ----------------------------
@torch.no_grad()
def _extract(loader):
    """Extract averaged-head L2-normalised features for a loader."""
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device)
        fs   = torch.stack(model.extract_all(imgs), dim=0).mean(0)
        feats.append(F.normalize(fs, p=2, dim=1).cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)

@torch.no_grad()
def evaluate():
    model.eval()
    for h in arc_heads:
        h.eval()

    # ── 1. Full similarity-based accuracy ───────────────────────────────────
    # All M test images serve as both gallery and query.
    # Cosine similarity matrix [M, M]; diagonal set to -1 to exclude self-match.
    # Rank-1: each image's nearest neighbour (excluding itself) is its prediction.
    # EER: all unique pairs (i<j) split into genuine / impostor by label equality.
    all_feats, all_labels = _extract(test_loader)
    M   = len(all_labels)
    sim = torch.mm(all_feats, all_feats.t())   # [M, M] cosine similarity
    sim.fill_diagonal_(-1.0)                   # exclude self-match

    pred_full = all_labels[sim.argmax(dim=1)]
    acc_full  = (pred_full == all_labels).float().mean().item() * 100

    sim_np = sim.numpy()
    gen_full, imp_full = [], []
    for i in range(M):
        for j in range(i + 1, M):          # upper triangle only (unique pairs)
            s = sim_np[i, j]
            (gen_full if all_labels[i] == all_labels[j] else imp_full).append(s)
    eer_full = compute_eer(gen_full, imp_full)

    # ── 2. Split similarity-based accuracy ──────────────────────────────────
    # Gallery: first image per identity (registered template).
    # Query  : all remaining images (probe set).
    # Similarity matrix [Q, G]; no diagonal issue — sets are disjoint.
    # Rank-1: each query matched to nearest gallery sample.
    # EER: all (query_i, gallery_j) pairs split by label equality.
    if len(query_samples) > 0:
        gal_feats, gal_labels = _extract(gallery_loader)   # [G, d]
        qry_feats, qry_labels = _extract(query_loader)     # [Q, d]

        sim_split = torch.mm(qry_feats, gal_feats.t())     # [Q, G]
        pred_split = gal_labels[sim_split.argmax(dim=1)]
        acc_split  = (pred_split == qry_labels).float().mean().item() * 100

        sim_split_np = sim_split.numpy()
        gen_split, imp_split = [], []
        for i in range(len(qry_labels)):
            for j in range(len(gal_labels)):
                s = sim_split_np[i, j]
                (gen_split if qry_labels[i] == gal_labels[j] else imp_split).append(s)
        eer_split = compute_eer(gen_split, imp_split)
    else:
        acc_split = eer_split = float("nan")

    # ── 3. ArcFace head classification accuracy ─────────────────────────────
    # Each ArcFace head has a weight matrix W: [num_total_classes, feature_dim].
    # The rows of W are learned class prototype vectors.
    # Classification score for class c = cosine_sim(feature, W[c])
    #                                  = feature @ W[c] / (||feature|| ||W[c]||)
    # Since extract() L2-normalises features, and we L2-normalise W rows,
    # this reduces to: logits = feature @ W.t()  (pure dot product on unit sphere).
    # argmax over classes → predicted identity integer.
    # Because all datasets share the same label map, predicted integer directly
    # matches test label integer — no remapping needed.
    #
    # Per-head accuracy: uses that head's feature extractor + that head's W.
    # Ensemble accuracy: average logits across all N heads before argmax.

    # Accumulate per-head logits over the full test set
    per_head_logits = [[] for _ in range(N)]
    all_test_labels = []

    for imgs, lbl in test_loader:
        imgs        = imgs.to(device)
        shared_feat = model.shared(imgs).view(imgs.size(0), -1)
        all_test_labels.append(lbl)
        for hi in range(N):
            feat = F.normalize(model.heads[hi](shared_feat), p=2, dim=1)  # [B, d]
            # L2-normalise W rows: W has shape [num_total_classes, feature_dim]
            W_norm = F.normalize(arc_heads[hi].W, p=2, dim=0).t()         # [num_total_classes, d]
            logits = feat @ W_norm.t()                                     # [B, num_total_classes]
            per_head_logits[hi].append(logits.cpu())

    all_test_labels = torch.cat(all_test_labels)   # [M]
    head_accs = []
    stacked_logits = []

    for hi in range(N):
        logits_hi = torch.cat(per_head_logits[hi], dim=0)   # [M, num_total_classes]
        stacked_logits.append(logits_hi)
        pred_hi  = logits_hi.argmax(dim=1)
        acc_hi   = (pred_hi == all_test_labels).float().mean().item() * 100
        head_accs.append(acc_hi)

    # Ensemble: average logits across all N heads, then argmax
    ens_logits  = torch.stack(stacked_logits, dim=0).mean(0)   # [M, num_total_classes]
    acc_arc_ens = (ens_logits.argmax(dim=1) == all_test_labels).float().mean().item() * 100

    # ── Print results ────────────────────────────────────────────────────────
    print(f"\n  ┌─ Evaluation on test domain '{test_domains[0]}' │ M={M} images")
    print(f"  │")
    print(f"  │  [1] Full similarity  "
          f"│ Rank-1: {acc_full:6.2f}%  EER: {eer_full:5.2f}%"
          f"  — all {M} images as gallery+query, self excluded")
    if not np.isnan(acc_split):
        print(f"  │  [2] Split similarity "
              f"│ Rank-1: {acc_split:6.2f}%  EER: {eer_split:5.2f}%"
              f"  — gallery={len(gallery_samples)}, query={len(query_samples)}")
    else:
        print(f"  │  [2] Split similarity │ N/A — each identity has only 1 image")
    print(f"  │")
    for hi, acc_hi in enumerate(head_accs):
        print(f"  │  [3] ArcFace head {hi}   "
              f"│ Acc: {acc_hi:6.2f}%"
              f"  — domain '{train_domains[hi]}' head, {num_total_classes} classes")
    print(f"  │  [3] ArcFace ensemble "
          f"│ Acc: {acc_arc_ens:6.2f}%"
          f"  — averaged logits over {N} heads")
    print(f"  └{'─'*65}\n")

    return acc_full, eer_full

# ----------------------------
# 10. Training
# ----------------------------

# ── Phase 1: Supervised pre-training (L_sup only) ────────────────────────────
print(f"{'='*60}")
print(f"  Phase 1 — Supervised Pre-training  ({pretrain_epochs} epochs)")
print(f"  L_sup on original + Fourier-augmented images per head")
print(f"{'='*60}")

for epoch in range(pretrain_epochs):
    model.train()
    for h in arc_heads:
        h.train()
    epoch_loss, epoch_corr, epoch_total = 0.0, 0, 0

    for _ in tqdm(range(steps_per_epoch),
                  desc=f"Pretrain {epoch+1}/{pretrain_epochs}", leave=False):
        batches = [
            (imgs.to(device), lbl.to(device))
            for imgs, lbl in (il.next() for il in inf_loaders)
        ]
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        for i, (src_imgs, src_lbl) in enumerate(batches):
            feat  = model.extract(src_imgs, i)
            loss += arc_heads[i](feat, src_lbl)

            with torch.no_grad():
                preds        = arc_heads[i].get_logits(feat).argmax(dim=1)
                epoch_corr  += (preds == src_lbl).sum().item()
                epoch_total += src_lbl.size(0)

            for j in range(N):
                if i == j:
                    continue
                sty, _ = batches[j]
                if sty.size(0) != src_imgs.size(0):
                    idx_ = torch.randint(sty.size(0), (src_imgs.size(0),), device=device)
                    sty  = sty[idx_]
                aug_feat = model.extract(fourier_augment_batch(src_imgs, sty, lam), i)
                loss    += arc_heads[i](aug_feat, src_lbl)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / steps_per_epoch
    avg_acc  = 100.0 * epoch_corr / epoch_total if epoch_total > 0 else 0.0
    print(f"  Epoch [{epoch+1}/{pretrain_epochs}]  "
          f"Loss: {avg_loss:.4f}  Train Acc: {avg_acc:.2f}%")

    if (epoch + 1) % eval_every == 0:
        evaluate()

# ── Phase 2: Full PDFG training (all losses, Eq. 11) ─────────────────────────
print(f"\n{'='*60}")
print(f"  Phase 2 — Full PDFG Training  ({epochs} epochs, Eq. 11)")
print(f"  L = L_sup + L_ada + α·L_con + β·L_d-t")
print(f"{'='*60}")

best_eer = float("inf")

for epoch in range(epochs):
    model.train()
    for h in arc_heads:
        h.train()
    log = {"total": 0., "sup": 0., "ada": 0., "con": 0., "dt": 0.}
    epoch_corr, epoch_total = 0, 0

    for _ in tqdm(range(steps_per_epoch),
                  desc=f"Train {epoch+1}/{epochs}", leave=False):
        batches = [
            (imgs.to(device), lbl.to(device))
            for imgs, lbl in (il.next() for il in inf_loaders)
        ]

        aug = {}
        for i in range(N):
            src_imgs, src_lbl = batches[i]
            for j in range(N):
                if i == j:
                    continue
                sty, _ = batches[j]
                if sty.size(0) != src_imgs.size(0):
                    idx_ = torch.randint(sty.size(0), (src_imgs.size(0),), device=device)
                    sty  = sty[idx_]
                aug[(i, j)] = (fourier_augment_batch(src_imgs, sty, lam), src_lbl)

        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        # ── L_sup ────────────────────────────────────────────────────────
        orig_feats = []
        for i, (src_imgs, src_lbl) in enumerate(batches):
            feat  = model.extract(src_imgs, i)
            l_sup = arc_heads[i](feat, src_lbl)
            orig_feats.append(feat)
            loss       += l_sup
            log["sup"] += l_sup.item()

            with torch.no_grad():
                preds        = arc_heads[i].get_logits(feat).argmax(dim=1)
                epoch_corr  += (preds == src_lbl).sum().item()
                epoch_total += src_lbl.size(0)

            for j in range(N):
                if i == j:
                    continue
                aug_imgs, aug_lbl = aug[(i, j)]
                l_sup_aug  = arc_heads[i](model.extract(aug_imgs, i), aug_lbl)
                loss       += l_sup_aug
                log["sup"] += l_sup_aug.item()

        # ── L_con + L_d-t ────────────────────────────────────────────────
        for i in range(N):
            aug_head_feats, aug_labels_list = [], []
            for j in range(N):
                if i == j:
                    continue
                aug_imgs, aug_lbl = aug[(i, j)]
                aug_head_feats.append(model.extract_all(aug_imgs))
                aug_labels_list.append(aug_lbl)

            l_con       = alpha * consistent_loss(orig_feats[i], aug_head_feats)
            loss       += l_con
            log["con"] += l_con.item()

            src_labels_i = batches[i][1]
            aug_avg      = torch.stack(
                [torch.stack(hf, 0).mean(0) for hf in aug_head_feats], 0
            ).mean(0)
            pos, neg = sample_triplet_pairs(aug_avg, aug_labels_list[0], src_labels_i)
            l_dt     = beta * triplet_loss_fn(orig_feats[i], pos, neg, triplet_margin)
            loss       += l_dt
            log["dt"]  += l_dt.item()

        # ── L_ada ────────────────────────────────────────────────────────
        for i in range(N):
            for j in range(i + 1, N):
                l_ada       = mkmmd_loss(orig_feats[i], orig_feats[j])
                loss       += l_ada
                log["ada"] += l_ada.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        log["total"] += loss.item()

    for k in log:
        log[k] /= steps_per_epoch
    avg_acc = 100.0 * epoch_corr / epoch_total if epoch_total > 0 else 0.0
    print(
        f"  Epoch [{epoch+1}/{epochs}]  Loss: {log['total']:.4f}  "
        f"Train Acc: {avg_acc:.2f}%  |  "
        f"sup={log['sup']:.4f}  ada={log['ada']:.4f}  "
        f"con={log['con']:.4f}  dt={log['dt']:.4f}"
    )

    if (epoch + 1) % eval_every == 0:
        acc_full, eer_full = evaluate()
        if not np.isnan(eer_full) and eer_full < best_eer:
            best_eer = eer_full
            torch.save(
                {
                    "model"    : model.state_dict(),
                    "arc_heads": [h.state_dict() for h in arc_heads],
                    "epoch"    : epoch + 1,
                },
                "best_model.pth",
            )
            print(f"  ✓ New best EER: {eer_full:.2f}%  -> saved best_model.pth")
        print()

print(f"\nDone. Best EER: {best_eer:.2f}%")
