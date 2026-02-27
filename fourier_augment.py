"""
Fourier-Based Data Augmentation for PDFG  (Section III-C, Eqs. 2–7)

Key idea:
  - Amplitude spectrum  → low-level style (illumination, device response)
  - Phase spectrum      → high-level semantics (palm lines, identity)

By mixing amplitude from one dataset with phase from another,
we generate new images with the same identity but a new visual style.

New image  x^{D1→D2}  has:
  - Same CLASS / identity as  x^D1
  - New STYLE similar to      x^D2

For N source datasets, this produces N×(N-1)/2 augmented dataset pairs.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List


def fourier_augment(
    img1: np.ndarray,
    img2: np.ndarray,
    lam: float = 0.8,
) -> np.ndarray:
    """
    Generate a new image by mixing the amplitude of img1 and img2.
    Implements Equations (2)–(7) from the paper.

    Args:
        img1: Source image array, shape [H, W] or [H, W, C], float32 in [0, 1]
        img2: Style donor image, same shape as img1
        lam:  Interpolation weight λ (default 0.8 per paper)
              New amplitude = (1-λ)·A(img1) + λ·A(img2)
              λ=0 keeps img1 style; λ=1 fully adopts img2 style

    Returns:
        Augmented image with same shape and dtype as img1
    """
    assert img1.shape == img2.shape, "img1 and img2 must have the same shape"

    if img1.ndim == 2:
        return _augment_single_channel(img1, img2, lam)
    else:
        # Process each channel independently
        channels = [
            _augment_single_channel(img1[..., c], img2[..., c], lam)
            for c in range(img1.shape[-1])
        ]
        return np.stack(channels, axis=-1)


def _augment_single_channel(
    ch1: np.ndarray,
    ch2: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Core Fourier augmentation for a single-channel image."""
    # Eq. (2): Fourier transform
    F1 = np.fft.fft2(ch1)
    F2 = np.fft.fft2(ch2)

    # Eq. (3): Amplitude components
    A1 = np.abs(F1)   # A(x^D1)
    A2 = np.abs(F2)   # A(x^D2)

    # Eq. (4): Phase of img1 (preserves identity/semantics)
    P1 = np.angle(F1)

    # Eq. (5): Mixed amplitude  Â(x^D1) = (1-λ)·A1 + λ·A2
    A_mixed = (1 - lam) * A1 + lam * A2

    # Eq. (6): Reconstruct F with mixed amplitude and original phase
    # F(x̂^D1)(u,v) = Â(x^D1)(u,v) · e^{-j·P(x^D1)(u,v)}
    F_new = A_mixed * np.exp(1j * P1)

    # Eq. (7): Inverse FFT to get augmented image
    img_new = np.real(np.fft.ifft2(F_new))

    # Clip to valid range
    img_new = np.clip(img_new, 0, 1).astype(np.float32)
    return img_new


# ---------------------------------------------------------------------------
# Batch-level augmentation utilities
# ---------------------------------------------------------------------------

def fourier_augment_batch(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    lam: float = 0.8,
) -> torch.Tensor:
    """
    Fourier augmentation applied to a batch of image tensors.

    Args:
        batch1: Images from dataset D1, shape [B, C, H, W], float in [0, 1]
        batch2: Images from dataset D2, same shape (style donors)
        lam:    Interpolation weight λ

    Returns:
        Augmented batch [B, C, H, W] — same class as batch1, new style from batch2
    """
    B, C, H, W = batch1.shape
    result = torch.zeros_like(batch1)

    b1_np = batch1.cpu().numpy()
    b2_np = batch2.cpu().numpy()

    for i in range(B):
        for c in range(C):
            result[i, c] = torch.from_numpy(
                _augment_single_channel(b1_np[i, c], b2_np[i, c], lam)
            )

    return result.to(batch1.device)


def generate_all_augmented_pairs(
    datasets: List[torch.Tensor],
    lam: float = 0.8,
) -> dict:
    """
    For N source datasets, generate all N×(N-1)/2 augmented dataset pairs.
    Paper says: "For N source datasets, N×N/2 new datasets can be augmented."

    Args:
        datasets: List of N tensors, each [B, C, H, W]
        lam:      Fourier interpolation weight

    Returns:
        Dict mapping (src_idx, style_idx) → augmented tensor
        e.g., {(0,1): aug_D0→D1, (1,0): aug_D1→D0, ...}
    """
    N = len(datasets)
    augmented = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                # For each image in D_i, randomly pair with an image in D_j
                src = datasets[i]
                style = datasets[j]
                # Randomly sample from style dataset if sizes differ
                if style.shape[0] != src.shape[0]:
                    idx = torch.randint(0, style.shape[0], (src.shape[0],))
                    style_sampled = style[idx]
                else:
                    style_sampled = style
                augmented[(i, j)] = fourier_augment_batch(src, style_sampled, lam)
    return augmented


# ---------------------------------------------------------------------------
# Visualisation helper (optional)
# ---------------------------------------------------------------------------

def visualize_augmentation(img1: np.ndarray, img2: np.ndarray, lam: float = 0.8):
    """
    Show original, style donor, and augmented image side-by-side.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    aug = fourier_augment(img1, img2, lam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap = "gray" if img1.ndim == 2 else None

    axes[0].imshow(img1, cmap=cmap); axes[0].set_title("Source (x^D1)")
    axes[1].imshow(img2, cmap=cmap); axes[1].set_title("Style donor (x^D2)")
    axes[2].imshow(aug, cmap=cmap);  axes[2].set_title(f"Augmented x^D1→D2 (λ={lam})")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
