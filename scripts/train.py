# ──────────────────────────────────────────────────────────────
# Training script — main entry point
# Extracted from optical_flow.ipynb Cells 8, 11, 13(exec), 19, 44, 51, 52
# ──────────────────────────────────────────────────────────────

# Cell 8 — Imports (adapted for repository structure)
import os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# Project imports
from configs.default import *
from src.model import OpticalFlowNet
from src.dataset import (
    discover_triplets,
    OpticalFlowAugmentor,
    FlyingChairsDataset,
    normalize_imgs,
)
from src.losses import multiscale_loss_with_photo
from src.viz import visualize_flow_comparison, plot_magnitude_histograms


# ── Cell 11: Seed + device ───────────────────────────────────

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
print(f"PyTorch: {torch.__version__}")


# ── Cell 13 (exec): Discover triplets + train/val split ─────

triplets = discover_triplets(DATA_ROOT, max_samples=SUBSET_SIZE)

n_val          = int(len(triplets) * VAL_SPLIT)
n_train        = len(triplets) - n_val
train_triplets = triplets[:n_train]
val_triplets   = triplets[n_train:]
print(f"Train: {n_train:,} | Val: {n_val:,}")


# ── Cell 19: Augmentor + DataLoaders ─────────────────────────

augmentor    = OpticalFlowAugmentor(output_size=RESOLUTION)                                             # Augmentor object
train_ds     = FlyingChairsDataset(train_triplets, augmentor=augmentor, resolution=RESOLUTION)          # Training dataset object
val_ds       = FlyingChairsDataset(val_triplets,   augmentor=None,      resolution=RESOLUTION)          # Validation dataset object

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=12,                                                                                    # Adjust according to your GPU ... this was set for A100 so you get the refference
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# ── Cell 44: Model instantiation + sanity check ─────────────

model = OpticalFlowNet(base_ch=BASE_CH, max_disp=MAX_DISP, n_iters=N_GRU_ITERS).to(device)
if hasattr(torch, 'compile'):                           # first compile run will be slower, but after that it should be much faster
    model = torch.compile(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params/1e6:.2f}M")

# Forward-pass sanity check
with torch.no_grad():
    _x = torch.randn(2, 3, RESOLUTION, RESOLUTION, device=device)
    _y = torch.randn(2, 3, RESOLUTION, RESOLUTION, device=device)
    fs2, fs1, ff = model(_x, _y)
    print(f"flow_s2  shape : {tuple(fs2.shape)}")                                       # expected (2, 2, 64,  64)
    print(f"flow_s1  shape : {tuple(fs1.shape)}")                                       # expected (2, 2, 128, 128)
    print(f"flow_full shape: {tuple(ff.shape)}")                                        # expected (2, 2, 256, 256)
    assert ff.shape == (2, 2, RESOLUTION, RESOLUTION), "Output resolution mismatch"
print("Sanity check passed")


# ── Cell 51: Training loop ───────────────────────────────────

"""
I get that this is not very production like, but
Please note that you will have to run this code block twice, once with the scratch values, 
Second time with the fine tuning values
the values are mentioned beside the relevant variables in comments
"""

scaler = GradScaler("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


"""
Load the previously trained model if exists
"""
_ckpt_path = os.path.join('models', f'{final_optic_flow_model_name}.pt')
if os.path.exists(_ckpt_path):
    _ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(_ckpt['model_state'])
        best_val_epe = _ckpt.get('val_epe', float('inf'))
        print(f'Loaded checkpoint: epoch={_ckpt["epoch"]}, val_epe={best_val_epe:.4f}')
        print('Starting Phase 2: fine-tuning with warping loss')
    except Exception as e:
        best_val_epe = float('inf')
        print('Architecture mismatch (probably H/4 update) training from scratch')
else:
    best_val_epe = float('inf')
    print('No checkpoint found training from scratch with warping loss')


# Fine-tuning scheduler: lower max_lr since model if 
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,                                            # for training from scratch : 3e-4,  fine-tuning phase = 1.5e-4
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,                                          # 0.1 to 0.2 for scratch training, 0.05 for fine tuning
    div_factor=10,
    final_div_factor=1000,
    anneal_strategy='cos'
)
train_history = {'loss': [], 'epe': []}
val_history   = {'loss': [], 'epe': []}

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    t_losses, t_epes = [], []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

    for step, (img1, img2, flow_gt) in enumerate(pbar, 1):
      img1, img2, flow_gt = img1.to(device), img2.to(device), flow_gt.to(device)

      optimizer.zero_grad()

      # Mixed precision forward
      with autocast("cuda"):
          pred_flows = model(normalize_imgs(img1), normalize_imgs(img2))
          loss, ep   = multiscale_loss_with_photo(
              pred_flows, flow_gt, img1, img2,
              weights=SCALE_WEIGHTS,
              smooth_w=SMOOTHNESS_W,
              grad_w=GRAD_W,
              photo_w=PHOTO_W,
          )

      #  Scaled backward
      scaler.scale(loss).backward()

      # Gradient clipping (needs unscale first)
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

      # Optimizer step
      scaler.step(optimizer)
      scaler.update()

      scheduler.step()

      t_losses.append(loss.item())
      t_epes.append(ep)

      pbar.set_postfix({
          "loss": f"{np.mean(t_losses[-20:]):.3f}",
          "epe":  f"{np.mean(t_epes[-20:]):.3f}",
          "lr":   f"{scheduler.get_last_lr()[0]:.1e}"
      })

    # Validation
    model.eval()
    v_losses, v_epes = [], []

    vbar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for img1, img2, flow_gt in vbar:
            img1, img2, flow_gt = img1.to(device), img2.to(device), flow_gt.to(device)

            with autocast("cuda"):
                pred_flows = model(normalize_imgs(img1), normalize_imgs(img2))
                # Use same loss as training for fair comparison
                loss, ep = multiscale_loss_with_photo(
                    pred_flows, flow_gt, img1, img2,
                    weights=SCALE_WEIGHTS,
                    smooth_w=SMOOTHNESS_W,
                    grad_w=GRAD_W,
                    photo_w=PHOTO_W,
                )

            v_losses.append(loss.item())
            v_epes.append(ep.item() if torch.is_tensor(ep) else ep)

            vbar.set_postfix({
                "val_loss": f"{np.mean(v_losses):.3f}",
                "val_epe":  f"{np.mean(v_epes):.3f}"
            })
    mtr_l, mtr_e = np.mean(t_losses), np.mean(t_epes)
    mvl_l, mvl_e = np.mean(v_losses), np.mean(v_epes)
    train_history['loss'].append(mtr_l); train_history['epe'].append(mtr_e)
    val_history['loss'].append(mvl_l);   val_history['epe'].append(mvl_e)

    print(f"Epoch {epoch:3d} | Train loss={mtr_l:.4f} epe={mtr_e:.4f} | ",
          f"Val loss={mvl_l:.4f} epe={mvl_e:.4f}")

    
    # Best model checkpoint
    if mvl_e < best_val_epe:
        best_val_epe = mvl_e
        torch.save({
            'epoch'       : epoch,
            'model_state' : model.state_dict(),
            'optim_state' : optimizer.state_dict(),
            'val_epe'     : best_val_epe,
            'config'      : {
                'BASE_CH'    : BASE_CH,
                'MAX_DISP'   : MAX_DISP,
                'N_GRU_ITERS': N_GRU_ITERS,
                'RESOLUTION' : RESOLUTION,
            },
        }, 'models/flow_model.pt')
        print(f"New best val EPE: {best_val_epe:.4f}  saved models/flow_model.pt")
        with torch.no_grad():
            _i1, _i2, _gt = next(iter(val_loader))
            _i1 = _i1.to(device, non_blocking=True)
            _i2 = _i2.to(device, non_blocking=True)
            _gt = _gt.to(device, non_blocking=True)
            _pred_flows = model(normalize_imgs(_i1), normalize_imgs(_i2))
            _ff = _pred_flows[-1]
            visualize_flow_comparison(
                _i1[0], _ff[0], _gt[0],
                tag=f"epoch{epoch:03d}"
            )
            plot_magnitude_histograms(_ff[0], _gt[0], tag=f"epoch{epoch:03d}")
print(f"\nDone. Best val EPE = {best_val_epe:.4f} px")


# ── Cell 52: Final checkpoint save ───────────────────────────

torch.save({
    'epoch'       : EPOCHS,
    'model_state' : model.state_dict(),
    'train_history': train_history,
    'val_history'  : val_history,
    'config'       : {
        'BASE_CH'    : BASE_CH,
        'MAX_DISP'   : MAX_DISP,
        'N_GRU_ITERS': N_GRU_ITERS,
        'RESOLUTION' : RESOLUTION,
        'BATCH_SIZE' : BATCH_SIZE,
        'EPOCHS'     : EPOCHS,
    },
}, f'models/{final_optic_flow_model_name}.pt')
print(f"Saved models/{final_optic_flow_model_name}.pt")
