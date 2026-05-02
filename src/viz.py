# ──────────────────────────────────────────────────────────────
# Visualisation utilities
# Extracted from optical_flow.ipynb Cell 46
# ──────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb


# ── Correct HSV flow colour (unchanged, already correct) ─────────────────────
def flow_to_color(flow_hw2: np.ndarray) -> np.ndarray:
    u, v   = flow_hw2[:, :, 0], flow_hw2[:, :, 1]
    mag    = np.sqrt(u ** 2 + v ** 2)
    angle  = np.arctan2(v, u)
    hue    = (angle / (2 * np.pi) + 0.5) % 1.0
    sat    = np.clip(mag / (mag.max() + 1e-6), 0, 1)
    val    = np.ones_like(hue)
    return (hsv_to_rgb(np.stack([hue, sat, val], -1).astype(np.float32)) * 255).astype(np.uint8)


# ── Shared-scale magnitude visualisation ─────────────────────────────────────
def visualize_flow_comparison(
    img1_t:      torch.Tensor,   # [3, H, W] uint8 or [0,1]
    pred_flow_t: torch.Tensor,   # [2, H, W]
    gt_flow_t:   torch.Tensor,   # [2, H, W]
    tag: str = "epoch000",
    out_dir: str = "./dia_images",
    cmap: str = "viridis",
):
    """
    Saves a 2×3 figure:
      Row 0: Input | Predicted flow (HSV) | Predicted magnitude
      Row 1: Error map (EPE) | GT flow (HSV) | GT magnitude
    GT and Pred magnitude share identical vmin/vmax so they're directly comparable.
    Error map uses its own scale anchored at 0.
    """
    os.makedirs(out_dir, exist_ok=True)

    img_np  = img1_t.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy()
    pred_np = pred_flow_t.permute(1, 2, 0).detach().cpu().numpy()   # H,W,2  float
    gt_np   = gt_flow_t.permute(1, 2, 0).cpu().numpy()

    # ── Magnitudes (float, NOT scaled to 255) ────────────────────────────────
    pred_mag = np.sqrt(pred_np[..., 0] ** 2 + pred_np[..., 1] ** 2)
    gt_mag   = np.sqrt(gt_np[..., 0]   ** 2 + gt_np[..., 1]   ** 2)

    # Shared scale: use GT max so clipping is semantically meaningful
    shared_vmax = float(gt_mag.max()) + 1e-6
    shared_vmin = 0.0

    # ── EPE per pixel ─────────────────────────────────────────────────────────
    diff   = pred_np - gt_np          # H,W,2
    epe_map = np.sqrt((diff ** 2).sum(-1))   # H,W  in pixels

    # ── HSV colour wheels ─────────────────────────────────────────────────────
    pred_rgb = flow_to_color(pred_np)
    gt_rgb   = flow_to_color(gt_np)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Flow diagnostics — {tag}", fontsize=12, fontweight="bold")

    def _show(ax, data, title, **kw):
        im = ax.imshow(data, **kw)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return im

    # Row 0
    _show(axes[0, 0], img_np,   "Input frame 1")
    _show(axes[0, 1], pred_rgb, "Predicted flow (HSV)")
    im_pm = _show(axes[0, 2], pred_mag, "Predicted magnitude (px)",
                  cmap=cmap, vmin=shared_vmin, vmax=shared_vmax)
    fig.colorbar(im_pm, ax=axes[0, 2], fraction=0.046, pad=0.04,
                 label="displacement (px)")

    # Row 1
    _show(axes[1, 0], epe_map,  "EPE per pixel (↑ = worse)",
          cmap="hot", vmin=0, vmax=epe_map.max())
    fig.colorbar(axes[1, 0].images[0], ax=axes[1, 0],
                 fraction=0.046, pad=0.04, label="EPE (px)")
    _show(axes[1, 1], gt_rgb,   "GT flow (HSV)")
    im_gm = _show(axes[1, 2], gt_mag, "GT magnitude (px)",
                  cmap=cmap, vmin=shared_vmin, vmax=shared_vmax)
    fig.colorbar(im_gm, ax=axes[1, 2], fraction=0.046, pad=0.04,
                 label="displacement (px)")

    mean_epe = epe_map.mean()
    fig.text(0.5, 0.01, f"Mean EPE = {mean_epe:.3f} px | "
             f"GT max = {gt_mag.max():.2f} px | Pred max = {pred_mag.max():.2f} px",
             ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    path = f"{out_dir}/diagnostic_{tag}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}  (mean EPE = {mean_epe:.3f} px)")
    return mean_epe


# ── Histogram comparison ──────────────────────────────────────────────────────
def plot_magnitude_histograms(pred_flow_t, gt_flow_t, tag="epoch000", out_dir="./train-histogram"):
    os.makedirs(out_dir, exist_ok=True)

    pred_np = pred_flow_t.permute(1, 2, 0).detach().cpu().numpy()
    gt_np   = gt_flow_t.permute(1, 2, 0).cpu().numpy()
    pred_mag = np.sqrt(pred_np[..., 0] ** 2 + pred_np[..., 1] ** 2).ravel()
    gt_mag   = np.sqrt(gt_np[..., 0]   ** 2 + gt_np[..., 1]   ** 2).ravel()

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(gt_mag.max(), pred_mag.max()) + 1, 60)
    ax.hist(gt_mag,   bins=bins, alpha=0.6, label="GT", color="#1f77b4")
    ax.hist(pred_mag, bins=bins, alpha=0.6, label="Predicted", color="#ff7f0e")
    ax.set_xlabel("Flow magnitude (pixels)"); ax.set_ylabel("Pixel count")
    ax.set_title(f"Magnitude distribution — {tag}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/hist_{tag}.png", dpi=100, bbox_inches="tight")
    plt.close()
