# Neural Optic-Flow

A RAFT-inspired neural optical flow estimation model trained on the [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) dataset.

## Architecture Overview

The pipeline follows a coarse-to-fine iterative refinement approach:

1. **Shared Encoder** — Extracts hierarchical features at H/2 and H/4 with skip connections
2. **Cost Volume** — Builds a correlation volume between frame features within a search window
3. **ConvGRU Refinement** — Iteratively refines the coarse flow estimate using recurrent updates
4. **Multi-scale Decoder** — Produces flow predictions at three scales (H/4, H/2, H) with attention-gated skip connections

### Loss Functions
- **EPE** (End-Point Error) — supervised flow error
- **Edge-aware Smoothness** — regularizes flow while preserving image edges
- **Gradient Consistency** — aligns spatial flow derivatives with ground truth
- **Photometric Reconstruction** — SSIM + Charbonnier L1 via backward warping (applied at finer scales)

## Project Structure

```
├── configs/
│   └── default.py          # All hyperparameters and configuration
├── src/
│   ├── model.py             # Network architecture
│   ├── dataset.py           # FlyingChairs data loading, .flo reader, augmentation
│   ├── losses.py            # Loss functions and backward warping utilities
│   └── viz.py               # Flow visualization and diagnostic plotting
├── scripts/
│   └── train.py             # Training entry point
├── models/                  # Pre-trained checkpoints
├── Notebooks/               # Original Jupyter notebooks
│   ├── optical_flow.ipynb   # Main training notebook
└── requirements.txt
```

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dataset
Download the [FlyingChairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) and place it in the project root as `FlyingChairs_release/` (or update `DATASET_LOCATION` in `configs/default.py`).

## Training

Run from the project root directory:

```bash
python -m scripts.train
```

### Configuration
All hyperparameters are in [`configs/default.py`](configs/default.py):
- `RESOLUTION` — Input resolution (default: 256)
- `BATCH_SIZE` — Training batch size (default: 32)
- `EPOCHS` — Number of training epochs (default: 60)
- `BASE_CH` — Base feature channels (default: 96)
- `MAX_DISP` — Cost volume search radius (default: 4)
- `N_GRU_ITERS` — ConvGRU refinement iterations (default: 6)

### Training Phases
The training script supports two-phase training:
1. **From scratch** — Use default `max_lr=3e-4`, `pct_start=0.1`
2. **Fine-tuning** — Adjust to `max_lr=1.5e-4`, `pct_start=0.05` in `scripts/train.py`

Checkpoints are saved to `models/flow_model.pt` (best validation EPE) and `models/{final_optic_flow_model_name}.pt` (final).

## Pre-trained Models
- `models/flow_model.pt` — Best validation checkpoint
- `models/final_optic_flow_model.pt` — Final trained model
