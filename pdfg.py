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
    At inference, features from all N heads are averaged then L2-normalised
    (paper Section III-B and Section IV-B).
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

    def extract_avg(self, x):
        """
        Paper evaluation protocol (Section IV-B):
            'features extracted by different feature extractors are averaged
             as the final feature and normalised by l2 normalisation.'
        Returns a single L2-normalised feature vector per image.
        """
        per_head = torch.stack(self.extract_all(x), dim=0)   # [N, B, d]
        avg      = per_head.mean(dim=0)                       # [B, d]
        return F.normalize(avg, p=2, dim=1)                   # [B, d]  ← final feature

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
all_hand_ids = set()
for fname in sorted(os.listdir(data_path)):
    if not fname.lower().endswith(".jpg"):
        continue
    parts = fname[:-4].split("_")
    if len(parts) != 4:
        continue
    subject_id, hand, spec, _ = parts
    if spec in set(train_domains):
        all_hand_ids.add(f"{subject_id}_{hand}")
shared_label_map  = {h: i for i, h in enumerate(sorted(all_hand_ids))}
num_total_classes = len(shared_label_map)
print(f"  Shared identity space: {num_total_classes} identities")

# ── Source datasets → registration set (gallery) ─────────────────────────────
# Paper: "The source datasets are used as the registration set"
src_datasets = [CASIASpectrum(data_path, [s], shared_label_map) for s in train_domains]

# ── Target dataset → query set ───────────────────────────────────────────────
# Paper: "the target dataset is used as the query set"
tgt_dataset  = CASIASpectrum(data_path, test_domains, shared_label_map)

train_loaders = [
    DataLoader(ds, batch_size=batch_size, shuffle=True,
               num_workers=0, pin_memory=False, drop_last=True)
    for ds in src_datasets
]

# Registration loader: all source domain images as gallery
# (combine all source domains into one loader for evaluation)
registration_dataset = torch.utils.data.ConcatDataset(src_datasets)
registration_loader  = DataLoader(registration_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)

# Query loader: all target domain images
query_loader = DataLoader(tgt_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)

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

print(f"  Source (registration): {train_domains}  —  {len(registration_dataset)} images")
print(f"  Target (query)        : {test_domains[0]}  —  {len(tgt_dataset)} images")
print(f"  Steps per epoch       : {steps_per_epoch}")
print(f"  Device                : {device}\n")

# ----------------------------
# 8. Model & Optimizer Setup
# ----------------------------
model = MultiDatasetExtractors(N, feature_dim).to(device)

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
def _extract_avg(loader):
    """
    Extract the averaged, L2-normalised feature for every image in `loader`.

    Paper (Section IV-B):
        'For each target image, the features extracted by different feature
         extractors are averaged as the final feature and normalised by l2
         normalisation.'

    The same averaging+normalisation is applied to registration (source)
    images so that both gallery and query live in the same feature space.
    """
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device)
        f    = model.extract_avg(imgs)   # [B, d]  averaged + L2-normalised
        feats.append(f.cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)   # [M, d], [M]


@torch.no_grad()
def evaluate():
    model.eval()
    for h in arc_heads:
        h.eval()

    # ── Extract features ─────────────────────────────────────────────────────
    # Registration set  = all source-domain images  (paper: "registration set")
    # Query set         = all target-domain images  (paper: "query set")
    reg_feats,  reg_labels  = _extract_avg(registration_loader)   # [G, d]
    qry_feats,  qry_labels  = _extract_avg(query_loader)          # [Q, d]

    G = len(reg_labels)
    Q = len(qry_labels)

    # ── Palmprint Identification (Rank-1 Accuracy) ───────────────────────────
    # Paper: "the image in query set is matched with the images in registration
    #         set to find the closest one. If they are belonging to the same
    #         subject, the matching is successful and the accuracy is calculated
    #         as the metric."
    #
    # Cosine similarity matrix: sim[q, g] = <qry_feats[q], reg_feats[g]>
    # Both feature sets are already L2-normalised → dot product = cosine sim.
    # For each query, find the gallery image with the highest cosine similarity.
    # No self-match issue: query set (target domain) ≠ registration set (source domain).
    sim = torch.mm(qry_feats, reg_feats.t())          # [Q, G]
    nn_idx   = sim.argmax(dim=1)                       # [Q] index into gallery
    pred_ids = reg_labels[nn_idx]                      # [Q] predicted identity
    acc      = (pred_ids == qry_labels).float().mean().item() * 100

    # ── Palmprint Verification (EER) ─────────────────────────────────────────
    # Paper: "the images of target dataset are matched with each other.
    #         The genuine matching from the same category and the imposter
    #         matching from the different categories are obtained.
    #         Then, Equal Error Rate (EER) is obtained as the metric."
    #
    # All (query_i, gallery_j) pairs are scored; genuine if same label.
    sim_np = sim.numpy()
    gen_scores, imp_scores = [], []
    for i in range(Q):
        for j in range(G):
            s = sim_np[i, j]
            if qry_labels[i] == reg_labels[j]:
                gen_scores.append(s)
            else:
                imp_scores.append(s)
    eer = compute_eer(gen_scores, imp_scores)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n  ┌─ Evaluation │ registration={train_domains} ({G} imgs) "
          f"│ query={test_domains[0]} ({Q} imgs)")
    print(f"  │  Feature  : averaged {N} heads → L2-normalised  (dim={feature_dim})")
    print(f"  │  Rank-1 Accuracy : {acc:6.2f}%")
    print(f"  │  EER             : {eer:5.2f}%")
    print(f"  └{'─'*65}\n")

    return acc, eer

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
        acc, eer = evaluate()
        if not np.isnan(eer) and eer < best_eer:
            best_eer = eer
            torch.save(
                {
                    "model"    : model.state_dict(),
                    "arc_heads": [h.state_dict() for h in arc_heads],
                    "epoch"    : epoch + 1,
                },
                "best_model.pth",
            )
            print(f"  ✓ New best EER: {eer:.2f}%  -> saved best_model.pth")
        print()

print(f"\nDone. Best EER: {best_eer:.2f}%")
