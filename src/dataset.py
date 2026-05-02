# ──────────────────────────────────────────────────────────────
# Dataset utilities — data loading, augmentation, preprocessing
# Extracted from optical_flow.ipynb Cells 13, 15, 16, 18
# ──────────────────────────────────────────────────────────────

import os
import struct
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ── Cell 13: .FLO reader + triplet discovery ─────────────────

# .FLO reader
def read_flo_file(path: str) -> np.ndarray:
    """Load .flo binary file into float32 (H, W, 2) array."""
    MAGIC = 202021.25
    fsize = os.path.getsize(path)
    if fsize < 12:
        raise ValueError(f"Corrupt .flo (too small: {fsize} bytes): {path}")
    with open(path, 'rb') as f:
        magic = struct.unpack('f', f.read(4))[0]
        if abs(magic - MAGIC) > 1e-3:
            raise ValueError(f"Bad .flo magic: {path}")
        w, h  = struct.unpack('ii', f.read(8))
        expected = h * w * 2 * 4
        data = f.read()
        if len(data) < expected:
            raise ValueError(f"Truncated .flo ({len(data)}/{expected} bytes): {path}")
        flow  = np.frombuffer(data, dtype=np.float32).reshape(h, w, 2)
    return flow



# Triplet discovery
def discover_triplets(root: str, max_samples: int = 8000, verbose: bool = True):
    data_dir = os.path.join(root, 'data') if os.path.isdir(os.path.join(root, 'data')) else root
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir!r}")

    all_files  = sorted(os.listdir(data_dir))
    base_names = sorted(
        {f.replace('_img1.ppm', '') for f in all_files if f.endswith('_img1.ppm')}
    )

    triplets = []
    for base in base_names:
        p1 = os.path.join(data_dir, base + '_img1.ppm')
        p2 = os.path.join(data_dir, base + '_img2.ppm')
        pf = os.path.join(data_dir, base + '_flow.flo')
        if os.path.exists(p1) and os.path.exists(p2) and os.path.exists(pf):
            triplets.append((base, p1, p2, pf))
        if len(triplets) >= max_samples:
            break

    if verbose:
        print(f"Valid triplets found : {len(triplets):,}  (cap={max_samples:,})")
    return triplets


# ── Cell 15: Augmentation class ──────────────────────────────

class OpticalFlowAugmentor:
    def __init__(self, output_size=256, do_hflip=True, do_vflip=False):
        self.output_size = output_size
        self.do_hflip = do_hflip
        self.do_vflip = do_vflip

    def __call__(self, img1, img2, flow):
        # img1, img2: numpy HWC (uint8)
        # flow: numpy HWC (float32)

        S = self.output_size
        h, w = img1.shape[:2]

        # random crop
        scale = np.random.uniform(0.7, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = np.random.randint(0, h - ch + 1)
        x0 = np.random.randint(0, w - cw + 1)

        img1 = img1[y0:y0+ch, x0:x0+cw]
        img2 = img2[y0:y0+ch, x0:x0+cw]
        flow = flow[y0:y0+ch, x0:x0+cw]

        # resize
        img1 = cv2.resize(img1, (S, S))
        img2 = cv2.resize(img2, (S, S))
        flow = cv2.resize(flow, (S, S))

        flow[..., 0] *= (S / cw)
        flow[..., 1] *= (S / ch)

        # flips
        if self.do_hflip and np.random.rand() < 0.5:
            img1 = img1[:, ::-1].copy()    # flip W idth axis
            img2 = img2[:, ::-1].copy()
            flow = flow[:, ::-1].copy()
            flow[..., 0] *= -1              # negate horizontal flow

        if self.do_vflip and np.random.rand() < 0.5:
            img1 = img1[::-1, :].copy()    # flip H eight axis
            img2 = img2[::-1, :].copy()
            flow = flow[::-1, :].copy()
            flow[..., 1] *= -1              # negate vertical flow

        # photometric augmentations
        # brightness / contrast
        if np.random.rand() < 0.8:
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.uniform(-20, 20)
            img1 = np.clip(img1 * alpha + beta, 0, 255)
            img2 = np.clip(img2 * alpha + beta, 0, 255)

        # gaussian blur
        if np.random.rand() < 0.3:
            k = np.random.choice([3, 5])
            img1 = cv2.GaussianBlur(img1, (k, k), 0)
            img2 = cv2.GaussianBlur(img2, (k, k), 0)

        # noise
        if np.random.rand() < 0.5:
            noise = np.random.randn(*img1.shape) * 5
            img1 = np.clip(img1 + noise, 0, 255)
            img2 = np.clip(img2 + noise, 0, 255)

        # to tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        return img1, img2, flow


# ── Cell 16: Image normalisation ─────────────────────────────

## Image Normalisation
#Resizing the tensorts using bilinear interpolation and rescale displacement vectors accordingly
def resize_flow_tensor(flow: torch.Tensor, h: int, w: int) -> torch.Tensor:
    _, _, H, W = flow.shape
    flow_r = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
    flow_r[:, 0] *= w / W
    flow_r[:, 1] *= h / H
    return flow_r


# ImageNet normalisation
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def normalize_imgs(img: torch.Tensor) -> torch.Tensor:
    m = _MEAN.to(img.device)
    s = _STD.to(img.device)
    return (img - m) / s


# ── Cell 18: Dataset class ───────────────────────────────────

class FlyingChairsDataset(Dataset):
    def __init__(self, triplets, augmentor=None, resolution=256):
        self.triplets   = triplets
        self.augmentor  = augmentor
        self.resolution = resolution

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        _, p1, p2, pf = self.triplets[idx]
    
        try:
            img1 = cv2.imread(p1, cv2.IMREAD_COLOR)
            img2 = cv2.imread(p2, cv2.IMREAD_COLOR)
    
            if img1 is None or img2 is None:
                raise ValueError(f"Failed to read image: {p1} or {p2}")
    
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
            flow = read_flo_file(pf)  # H x W x 2
    
        except (ValueError, OSError) as e:
            # Skip corrupt sample → move to next index
            return self.__getitem__((idx + 1) % len(self))
    
        # Augmentation path
        if self.augmentor is not None:
            return self.augmentor(img1, img2, flow)
    
        # Resize + scale flow
        S = self.resolution
        oh, ow = flow.shape[:2]
    
        img1 = cv2.resize(img1, (S, S), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (S, S), interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, (S, S), interpolation=cv2.INTER_LINEAR)
    
        flow[..., 0] *= (S / ow)
        flow[..., 1] *= (S / oh)
    
        # To tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
    
        return img1, img2, flow
