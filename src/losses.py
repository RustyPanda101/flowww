# ──────────────────────────────────────────────────────────────
# Losses & warping utilities
# Extracted from optical_flow.ipynb Cells 37, 39, 41, 48, 43
# ──────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F

from src.dataset import resize_flow_tensor


# ── Cell 37: Backward warping ────────────────────────────────

"""
Backward-warp `feat` using `flow`.
feat : [B, C, H, W]
flow : [B, 2, H, W] flow in PIXELS at the feat resolution
Data flow:
    feat + flow → build coordinate grid
                → add flow (pixel displacement)
                → normalize to [-1, 1]
                → grid_sample (bilinear interpolation)

Output:
    warped feature map aligned using flow
"""
def warp_features(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape

    # Build normalised sampling grid [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=feat.dtype, device=feat.device),
        torch.arange(W, dtype=feat.dtype, device=feat.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)                # [1, 2, H, W]
    grid = grid.expand(B, -1, -1, -1)

    # Add flow and normalise to [-1, 1]
    sample_x = (grid[:, 0] + flow[:, 0]) / (W - 1) * 2 - 1                  # [B, H, W]
    sample_y = (grid[:, 1] + flow[:, 1]) / (H - 1) * 2 - 1

    # expects [B, H, W, 2]
    sample_grid = torch.stack([sample_x, sample_y], dim=-1)

    return F.grid_sample(
        feat, sample_grid,
        mode="bilinear",
        padding_mode="border",                                              # border avoids black edges at frame boundaries
        align_corners=True,
    )


def warp_image(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return warp_features(img, flow)


# ── Cell 39: SSIM helper ─────────────────────────────────────

"""
Computes a per-pixel Structural Similarity Map(SSIM) between two images
Computes:
       luminance (mean)
       contrast (variance)
       structure (covariance)

"""
def _ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 3) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window // 2
    
    # luminance (mean)
    mu_x  = F.avg_pool2d(x, window, stride=1, padding=pad)
    mu_y  = F.avg_pool2d(y, window, stride=1, padding=pad)
    mu_xy = mu_x * mu_y
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2

    # Variances (contrast)
    sig_x2  = F.avg_pool2d(x * x, window, stride=1, padding=pad) - mu_x2
    sig_y2  = F.avg_pool2d(y * y, window, stride=1, padding=pad) - mu_y2

    # Covariance (structure)
    sig_xy  = F.avg_pool2d(x * y, window, stride=1, padding=pad) - mu_xy

    # ((2 * μ_x * μ_y + C1) * (2 * σ_xy + C2)) / ((μ_x² + μ_y² + C1) * (σx² + σy² + C2))
    ssim = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
           ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    
    return ssim.clamp(0, 1)


# ── Cell 41: Photometric loss ────────────────────────────────

"""
Photometric reconstruction loss for optical flow.

Given flow from img1 → img2:
    - Warp img2 back into img1's frame
    - Compare reconstructed image with img1

Loss = alpha * SSIM + (1 - alpha) * Charbonnier L1

Notes:
    - Only one image is warped (img2). img1 is the reference.
    - Mask removes invalid (out-of-bounds) warped pixels.
    - SSIM captures structural similarity; L1 captures pixel accuracy.
"""
def photometric_loss(
    img1:         torch.Tensor,   # [B, 3, H, W] unnormalised [0,1]
    img2:         torch.Tensor,   # [B, 3, H, W]
    flow_pred:    torch.Tensor,   # [B, 2, H, W]
    alpha:        float = 0.85,   # SSIM vs L1 blend
) -> torch.Tensor:

    img2_warped = warp_image(img2, flow_pred)                                           # Warp img2 into img1 frame using predicted flow

    valid = ((img2_warped > 0) & (img2_warped < 1)).all(dim=1, keepdim=True).float()    # keeps only pixels that map inside image bounds

    l1 = ((img2_warped - img2) ** 2 + 1e-4).sqrt()                                      # Charbonnier L1

                                                                                        
    ssim_map = _ssim_map(img2_warped, img2)                                             # SSIM loss (1 - SSIM)
    ssim_loss = (1.0 - ssim_map) / 2.0                                                  # normalize to [0, 1]

    # Combine losses: alpha * SSIM + (1-alpha) * L1
    photo = alpha * ssim_loss.mean(dim=1, keepdim=True) + (1.0 - alpha) * l1.mean(dim=1, keepdim=True)

    loss = (valid * photo).sum() / (valid.sum() + 1e-6)                                # Masked average loss
    return loss


# ── Cell 48: EPE + edge-aware smoothness + multiscale_loss ───

"""
Mean end-point error with small epsilon for numerical stability
"""
def epe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((pred - gt) ** 2, dim=1) + 1e-6).mean()

"""
penalises flow gradients, weighted by image gradients so edges are preserved
"""
def edge_aware_smoothness(flow: torch.Tensor, img: torch.Tensor, alpha: float = 150.0) -> torch.Tensor:
    
    # Flow spatial gradients
    fd_x = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
    fd_y = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()
    # Image edge weights (grayscale)
    gray = img.mean(dim=1, keepdim=True)
    id_x = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
    id_y = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
    w_x  = torch.exp(-alpha * id_x)
    w_y  = torch.exp(-alpha * id_y)
    return (w_x * fd_x).mean() + (w_y * fd_y).mean()


def multiscale_loss(
    pred_flows : tuple,
    flow_gt    : torch.Tensor,
    img1       : torch.Tensor,
    weights    : list  = (0.10, 0.30, 1.00),
    smooth_w   : float = 0.01,
    grad_w     : float = 0.1
) -> tuple:

    total    = flow_gt.new_zeros(1)
    epe_full = 0.0
    for i, (pred, w) in enumerate(zip(pred_flows, weights)):
        _, _, h, ww = pred.shape

        # Resize GT + image
        gt_r  = resize_flow_tensor(flow_gt, h, ww)
        img_r = F.interpolate(img1, (h, ww), mode='bilinear', align_corners=False)

        # Base losses
        l_epe = epe(pred, gt_r)
        l_smt = edge_aware_smoothness(pred, img_r)

        # Gradient loss
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        dx_gt = gt_r[:, :, :, 1:] - gt_r[:, :, :, :-1]
        dy_gt = gt_r[:, :, 1:, :] - gt_r[:, :, :-1, :]

        l_grad = (dx_pred - dx_gt).abs().mean() + (dy_pred - dy_gt).abs().mean()

        # Total
        total = total + w * (l_epe + smooth_w * l_smt + grad_w * l_grad)

        # Log full-res EPE
        if i == len(pred_flows) - 1:
            epe_full = l_epe.item()

    return total, epe_full


# ── Cell 43: multiscale_loss_with_photo ──────────────────────

"""
Multi-scale loss with EPE + smoothness + gradient + photometric (SSIM+L1).

- EPE: supervised flow error
- Smoothness: encourages spatial consistency (edge-aware)
- Gradient: preserves motion boundaries
- Photometric: enforces image reconstruction via warping (img2 → img1)

Photometric loss is applied only at finer scales (H/2, H) where flow is reliable.
"""

def multiscale_loss_with_photo(
    pred_flows: tuple,                              # (flow_s2, flow_s1, flow_full)
    flow_gt:    torch.Tensor,
    img1:       torch.Tensor,
    img2:       torch.Tensor,                       
    weights:    list  = (1.00, 0.50, 0.25),
    smooth_w:   float = 0.0001,
    grad_w:     float = 0.02,
    photo_w:    float = 0.10,
) -> tuple:

    total = flow_gt.new_zeros(1)
    epe_full = 0.0

    for i, (pred, w) in enumerate(zip(pred_flows, weights)):
        _, _, h, ww = pred.shape

        # Resize GT flow and img1 to current prediction scale
        gt_r = resize_flow_tensor(flow_gt, h, ww)
        img_r = F.interpolate(img1, (h, ww), mode='bilinear', align_corners=False)

        # EPE
        l_epe = epe(pred, gt_r)

        # Edge-aware smoothness
        l_smt = edge_aware_smoothness(pred, img_r)        # defined later

        # Gradient consistency: aligns spatial flow changes with GT (preserves edges)
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_gt = gt_r[:, :, :, 1:] - gt_r[:, :, :, :-1]
        dy_gt = gt_r[:, :, 1:, :] - gt_r[:, :, :-1, :]
        l_grad = (dx_pred - dx_gt).abs().mean() + (dy_pred - dy_gt).abs().mean()

        # sum supervised losses
        total = total + w * (l_epe + smooth_w * l_smt + grad_w * l_grad)

        # Photometric loss at finest 2 scales ... Skip coarsest scale (flow too rough for meaningful warping)
        if i >= 1:                                                                      # flow_s1 (H/2) and flow_full (H)
            img2_r = F.interpolate(img2, (h, ww), mode='bilinear', align_corners=False)
            l_photo = photometric_loss(img_r, img2_r, pred)
            total = total + w * photo_w * l_photo

        # Log full-res EPE
        if i == len(pred_flows) - 1:
            epe_full = l_epe.item()

    return total.squeeze(), epe_full
