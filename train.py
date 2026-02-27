"""
train.py  —  Entry point for PDFG on CASIA Multi-Spectral Palmprint dataset.

Usage:
  # Single experiment (e.g., sources=460,630 target=700)
  python train.py --root /path/to/CASIA-MS --sources 460 630 --target 700

  # Run all paper experiments from Table II (all source/target combinations)
  python train.py --root /path/to/CASIA-MS --run_all

  # Resume from checkpoint
  python train.py --root /path/to/CASIA-MS --sources 460 630 --target 700 --resume

Expected folder layout:
  /path/to/CASIA-MS/
  ├── 460/
  │   ├── 001_l_460_01.jpg   ← flat layout (subfolder_mode=False, default)
  │   └── ...
  ├── 630/
  ...
  OR
  ├── 460/
  │   ├── 001/               ← subfolder layout (use --subfolder_mode)
  │   │   └── img.jpg
  ...
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch

from dataset import CASIAMultiSpectral, CASIA_SPECTRA
from feature_extractor import MultiDatasetExtractors
from losses.losses import PDFGLoss
from trainer import PDFGTrainer


# ─────────────────────────────────────────────────────────────────────────────
#  Paper experiment combos from Table II
# ─────────────────────────────────────────────────────────────────────────────

# 2-source experiments (Table II rows with 2 sources)
TWO_SOURCE_COMBOS = [
    (["460", "630"],       "700"),
    (["460", "630"],       "850"),
    (["460", "630"],       "940"),
    (["460", "WHT"],       "630"),
    (["460", "WHT"],       "700"),
    (["460", "WHT"],       "850"),
    (["460", "700"],       "630"),
    (["460", "700"],       "850"),
    (["460", "700"],       "940"),
    (["630", "850"],       "460"),
    (["630", "850"],       "700"),
    (["630", "850"],       "WHT"),
    (["630", "700"],       "WHT"),
    (["630", "700"],       "850"),
    (["630", "700"],       "940"),
    (["460", "850"],       "630"),
    (["460", "850"],       "700"),
    (["460", "850"],       "940"),
    (["630", "WHT"],       "850"),
    (["630", "850"],       "WHT"),
    (["630", "850"],       "WHT"),
]

# 3-source experiments (Table II rows with 3 sources)
THREE_SOURCE_COMBOS = [
    (["460", "630", "700"], "850"),
    (["460", "630", "700"], "940"),
    (["460", "700", "WHT"], "630"),
    (["460", "700", "WHT"], "850"),
    (["460", "WHT"],        "630"),   # 2-source
    (["460", "630"],        "850"),   # 2-source
    (["630", "700", "WHT"], "460"),
    (["630", "700", "WHT"], "850"),
    (["630", "700", "WHT"], "940"),
    (["460", "850", "WHT"], "630"),
    (["460", "850", "WHT"], "700"),
    (["460", "850", "WHT"], "940"),
    (["630", "700", "WHT"], "460"),
    (["630", "850", "WHT"], "460"),
    (["630", "850", "WHT"], "700"),
    (["630", "850", "WHT"], "940"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args, sources, target, run_id=""):
    """Execute one CDPR-UT experiment."""
    tag = f"{'_'.join(sources)}→{target}" + (f"_{run_id}" if run_id else "")
    print(f"\n{'█'*60}")
    print(f"  Experiment: sources={sources}  target={target}")
    print(f"{'█'*60}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    dm = CASIAMultiSpectral(
        data_path       = args.root,
        source_spectra  = sources,
        target_spectrum = target,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seed            = args.seed,
    )
    dm.summary()

    # ── Model ─────────────────────────────────────────────────────────
    model = MultiDatasetExtractors(
        n_datasets   = len(sources),
        input_size   = 224,          # 224×224 RGB (matches your existing code)
        feature_dim  = args.feature_dim,
    )

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = PDFGLoss(
        num_classes_per_dataset=dm.num_classes_per_src,
        feature_dim=args.feature_dim,
        alpha=args.alpha,
        beta=args.beta,
        s=args.arcface_s,
        m=args.arcface_m,
    )

    # ── DataLoaders ───────────────────────────────────────────────────
    src_loaders  = dm.source_train_loaders()
    reg_loader   = dm.registration_loader()
    tgt_loader   = dm.target_loader()

    # Steps per epoch = length of the smallest source loader
    steps_per_epoch = args.steps_per_epoch or min(len(ld) for ld in src_loaders)

    # ── Trainer ───────────────────────────────────────────────────────
    save_dir = Path(args.save_dir) / tag
    trainer = PDFGTrainer(
        model=model,
        loss_fn=loss_fn,
        source_loaders=src_loaders,
        registration_loader=reg_loader,
        target_loader=tgt_loader,
        device=device,
        lr=args.lr,
        lam=args.lam,
        save_dir=str(save_dir),
        eval_every=args.eval_every,
        pretrain_epochs=args.pretrain_epochs,
    )

    if args.resume and (save_dir / "best_model.pth").exists():
        trainer.load("best_model.pth")

    # ── Train ─────────────────────────────────────────────────────────
    history = trainer.train(epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    # ── Final evaluation ──────────────────────────────────────────────
    print(f"\n  ── Final Evaluation (best checkpoint) ──")
    trainer.load("best_model.pth")
    trainer.model.eval()
    from utils.metrics import evaluate
    acc, eer = evaluate(trainer.model, reg_loader, tgt_loader, device, verbose=True)

    result = {
        "experiment": tag,
        "sources": sources,
        "target": target,
        "best_acc": acc,
        "best_eer": eer,
        "history": history,
    }

    # Save result JSON
    result_path = save_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"}, f, indent=2)

    print(f"\n  Results saved to {result_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PDFG: Cross-Dataset Palmprint Recognition with Unseen Target",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ──────────────────────────────────────────────────────────
    p.add_argument("--root",     default="/home/pai-ng/Jamal/CASIA-MS-ROI",
                                 help="Flat folder containing all .jpg ROI files")
    p.add_argument("--save_dir", default="runs", help="Directory for checkpoints & results")

    # ── Dataset ────────────────────────────────────────────────────────
    p.add_argument("--sources", nargs="+", default=["460", "700", "630"],
                   help="Source spectrum names (your train_domains)")
    p.add_argument("--target",  type=str,  default="940",
                   help="Target spectrum name  (your test_domains[0])")
    p.add_argument("--run_all",      action="store_true", help="Run all Table II experiments")
    p.add_argument("--batch_size",   type=int, default=8,       help="Mini-batch size (8 per paper)")
    p.add_argument("--num_workers",  type=int, default=2,       help="DataLoader workers")
    p.add_argument("--seed",         type=int, default=42,      help="Random seed")

    # ── Model ──────────────────────────────────────────────────────────
    p.add_argument("--feature_dim",  type=int,   default=128,   help="Output feature dimension")

    # ── Loss hyperparameters (paper defaults) ──────────────────────────
    p.add_argument("--alpha",        type=float, default=0.1,   help="L_con weight α")
    p.add_argument("--beta",         type=float, default=1.0,   help="L_d-t weight β")
    p.add_argument("--arcface_s",    type=float, default=64.0,  help="ArcFace scale s")
    p.add_argument("--arcface_m",    type=float, default=0.5,   help="ArcFace margin m")
    p.add_argument("--lam",          type=float, default=0.8,   help="Fourier augmentation λ")

    # ── Training ───────────────────────────────────────────────────────
    p.add_argument("--pretrain_epochs",  type=int,   default=30,   help="Phase 1: supervised pre-training epochs (L_sup only)")
    p.add_argument("--epochs",           type=int,   default=100,  help="Phase 2: full PDFG training epochs (all losses)")
    p.add_argument("--steps_per_epoch",  type=int,   default=None,
                   help="Steps per epoch for both phases (default: min loader length)")
    p.add_argument("--lr",               type=float, default=1e-4, help="Learning rate")
    p.add_argument("--eval_every",       type=int,   default=5,    help="Evaluate every N epochs (Phase 2 only)")
    p.add_argument("--resume",           action="store_true",       help="Resume from checkpoint")
    p.add_argument("--cpu",              action="store_true",       help="Force CPU (for debugging)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.run_all:
        # Reproduce all Table II experiments
        all_combos = TWO_SOURCE_COMBOS + THREE_SOURCE_COMBOS
        all_results = []
        for i, (sources, target) in enumerate(all_combos):
            result = run_experiment(args, sources, target, run_id=str(i))
            all_results.append(result)

        # Print summary table
        print(f"\n{'═'*65}")
        print(f"  {'Sources':<25} {'Target':<8} {'Acc (%)':>8} {'EER (%)':>8}")
        print(f"{'─'*65}")
        accs, eers = [], []
        for r in all_results:
            src_str = ",".join(r["sources"])
            print(f"  {src_str:<25} {r['target']:<8} {r['best_acc']:>8.2f} {r['best_eer']:>8.2f}")
            accs.append(r["best_acc"])
            eers.append(r["best_eer"])
        print(f"{'─'*65}")
        print(f"  {'Average':<25} {'':<8} {sum(accs)/len(accs):>8.2f} {sum(eers)/len(eers):>8.2f}")
        print(f"{'═'*65}")

        # Save all results
        summary_path = Path(args.save_dir) / "all_results.json"
        with open(summary_path, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "history"}
                       for r in all_results], f, indent=2)
        print(f"\n  All results saved to {summary_path}")

    else:
        if not args.sources or not args.target:
            print("ERROR: Provide --sources and --target, or use --run_all")
            print("Example: python train.py --root ./CASIA-MS --sources 460 630 --target 700")
            sys.exit(1)

        run_experiment(args, args.sources, args.target)


if __name__ == "__main__":
    main()
