# ──────────────────────────────────────────────────────────────
# Neural network model definitions
# Extracted from optical_flow.ipynb Cells 22, 24, 26, 28, 30, 32, 34
# ──────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import warp_features


# ── Cell 22: Residual Block ──────────────────────────────────

"""
a standard residual (skip-connection) block for refining features without losing original information
allows block to learn residual mapping (difference from input) instead of full transformation
helps preserve low-level spatial details

Data flow:
    Conv - > BatchNorm -> Relu -> Conv - > BatchNorm
                                                     }->  Relu ................. (Both convolution layers are 3x3)
                                    Skip Connections 
"""
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


# ── Cell 24: Encoder ─────────────────────────────────────────

"""
Encoder extracts hierarchical features and downsamples the image to a coarse representation (H/4) while preserving intermediate features for skip connections.
Data flow:
    x → block1 → s1
      → block2 → s2
      → block3 → c
      → block4 → block5 → f

Returns:
    f  : coarse features (H/4, 2*base_ch)
    s2 : skip features (H/4, base_ch)
    s1 : skip features (H/2, base_ch)

"""
class Encoder(nn.Module):
   
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        
        super().__init__()
        
        def _layer(inc, outc, stride, k):                                               # Everylayer is : Convolutional layer, with BatchNorm and Relu
            return nn.Sequential(
                nn.Conv2d(inc, outc, k, stride=stride, padding=k//2, bias=False),
                nn.BatchNorm2d(outc), nn.ReLU(inplace=True))

        self.block1 = _layer(in_ch, base_ch, stride=2, k=7)                             # convolution layer .. Kernel size 7x7, stride 2 , downsamples to H/2
        self.block2 = _layer(base_ch, base_ch, stride=2, k=5)                           # convolution layer .. Kernel size 5x5, stride 2 , downsamples to H/4
        self.block3 = _layer(base_ch, base_ch*2, stride=1, k=3)                         # no downsampling .. increase channels from 64 to 128 and produce coarse feature map

        self.block4 = nn.Sequential(                                                    # Refine the features at H/4
            _layer(base_ch*2, base_ch*2, stride=1, k=3),
            ResidualBlock(base_ch*2),                                                   # Preseve spatial information while improving representaion
        )

        self.block5 = nn.Sequential(
            ResidualBlock(base_ch * 2),
            ResidualBlock(base_ch * 2),
        )


    def forward(self, x):
        s1 = self.block1(x)                                                             # [B, 64,  H/2, W/2]
        s2 = self.block2(s1)                                                            # [B, 64,  H/4, W/4]
        c  = self.block3(s2)                                                            # [B, 128, H/4, W/4]
        f = self.block5(self.block4(c))                                                 # [B, 128, H/4, W/4]
        return f, s2, s1                    # coarse, skip from H/4, skip from H/2


# ── Cell 26: Cost Volume ─────────────────────────────────────

"""
Builds a correlation volume between two feature maps (f1, f2) by comparing each pixel in f1 with a neighborhood in f2.
Data flow:
    f1, f2 → L2 normalize
           → pad f2
           → for each (dy, dx) in search window:
                 shift f2 → compute dot product with f1
           → stack all correlations → cost volume

Output:
    corr : [B, (2*max_disp+1)^2, H, W]

Output: [B, (2·max_disp+1)^2, H, W]
"""
class CostVolume(nn.Module):
    def __init__(self, max_disp: int = 4, stride: int = 1, dilation: int = 1):
        super().__init__()
        
        self.max_disp = max_disp
        self.stride   = stride
        self.dilation = dilation
        
        # Pre-compute displacement offsets
        disps = list(range(-max_disp, max_disp + 1, stride))
        self._offsets = [(dy, dx) for dy in disps for dx in disps]

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        f1 = F.normalize(f1, p=2, dim=1)                                        # normalise f1
        f2 = F.normalize(f2, p=2, dim=1)                                        # normalise f2
        B, C, H, W = f1.shape                                                   # batch size, channels, heigth, width
        pad = self.max_disp * self.dilation
        f2p = F.pad(f2, [pad] * 4)
        corrs = []                                                              # corelation maps
        for dy, dx in self._offsets:
            f2_shifted = f2p[:, :, pad + dy*self.dilation : pad + dy*self.dilation + H,
                                   pad + dx*self.dilation : pad + dx*self.dilation + W]
            corrs.append((f1 * f2_shifted).sum(dim=1))

        corr = torch.stack(corrs, dim=1)                                        # stacking into the final cost volume
        return corr

"""
Compress wide cost-volume channel dimension to compact feature before GRU processing
Data flow:
    cost volume → Conv → BN → ReLU
                → Conv → BN → ReLU

Output:
    compressed feature map with reduced channels
"""
class CostVolumeEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        mid = max(in_ch // 2, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid,    3, padding=1, bias=False), nn.BatchNorm2d(mid),    nn.ReLU(inplace=True),
            nn.Conv2d(mid,   out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ── Cell 28: ConvGRU ─────────────────────────────────────────

"""
a recurrent unit that refines spatial features iteratively while preserving spatial structure.

"""
class ConvGRUCell(nn.Module):
    def __init__(self, input_ch: int, hidden_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        combined = input_ch + hidden_ch
        self.update = nn.Conv2d(combined, hidden_ch, kernel, padding=pad)
        self.reset  = nn.Conv2d(combined, hidden_ch, kernel, padding=pad)
        self.cand   = nn.Conv2d(combined, hidden_ch, kernel, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        if h is None:
            h = x.new_zeros(x.size(0), self.hidden_ch, x.size(2), x.size(3))
        xh = torch.cat([x, h], dim=1)
        z  = torch.sigmoid(self.update(xh))                                         # update gate
        r  = torch.sigmoid(self.reset(xh))                                          # reset gate
        q  = torch.tanh(self.cand(torch.cat([x, r * h], dim=1)))                    # candidate state
        return (1 - z) * h + z * q


# ── Cell 30: Decoder ─────────────────────────────────────────

"""
Decoder with skip connections from img1's encoder.
Produces flow at three scales for multi-scale supervision.

Returns (flow_s2, flow_s1, flow_full) at (H/4, H/2, H) respectively.
"""
class FlowDecoder(nn.Module):
    def __init__(self, gru_ch: int = 64, skip2_ch: int = 64, skip1_ch: int = 64):
        super().__init__()
        def _block(inc, outc):
            return nn.Sequential(
                nn.Conv2d(inc, outc, 3, padding=1, bias=False),
                nn.BatchNorm2d(outc), nn.ReLU(inplace=True))

        # H/4 -> H/4 refine
        self.up1     = _block(gru_ch, 64)
        self.fuse1   = _block(64 + skip2_ch,64)
        self.head_s2 = nn.Conv2d(64, 2, 3, padding=1)

        # H/4 -> H/2
        self.up2     = _block(64, 32)
        self.fuse2   = _block(32 + skip1_ch, 32)
        self.head_s1 = nn.Conv2d(32, 2, 3, padding=1)

        # H/2 -> H
        self.up3       = _block(32, 16)
        self.head_full = nn.Conv2d(16, 2, 3, padding=1)

    def forward(self, h, skip2, skip1):
        x = self.up1(h)                                                                         # GRU hidden → 64
        x = self.fuse1(torch.cat([x, skip2], dim=1))                                            # cat skip2
        flow_s2 = self.head_s2(x)                                                               # flow at h/4

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)              # increment to H/2 using bilinear inter 
        x = self.fuse2(torch.cat([self.up2(x), skip1], dim=1))                                  # cat skip1
        flow_s1 = self.head_s1(x)                                                               # flow at h/2

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        flow_full = self.head_full(self.up3(x))

        return flow_s2, flow_s1, flow_full


# ── Cell 32: OpticalFlowNet ──────────────────────────────────

# if you have followed comments till now, this should be pretty self explainatory

class OpticalFlowNet(nn.Module):
    def __init__(self, base_ch: int = 64, max_disp: int = 8, n_iters: int = 6):
        
        super().__init__()
        n_cv_ch       = (2 * max_disp + 1) ** 2
        self.encoder  = Encoder(in_ch=3, base_ch=base_ch)
        self.cost_vol = CostVolume(max_disp=max_disp)
        self.cv_enc   = CostVolumeEncoder(in_ch=n_cv_ch, out_ch=base_ch)
        
        # flow_enc compresses current flow estimate into feature
        self.flow_enc = nn.Sequential(
            nn.Conv2d(2, base_ch // 2, 7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        
        # GRU input: f1(2*base_ch) + cost_volume(base_ch) + flow(base_ch//2)
        self.gru     = ConvGRUCell(input_ch=base_ch * 2 + base_ch // 2 + base_ch,
                                   hidden_ch=base_ch)
        self.decoder = FlowDecoder(gru_ch=base_ch, skip2_ch=base_ch, skip1_ch=base_ch)
        self.n_iters = n_iters
        
        # Coarse flow initialiser (1x1 conv from GRU hidden)
        self.flow_head_coarse = nn.Conv2d(base_ch, 2, 1)

    def forward(self, img1, img2):
        f1, s2, s1 = self.encoder(img1)
        f2, _,  _  = self.encoder(img2)

        B, C, Hc, Wc = f1.shape

        # Initialise flow at coarse scale (H/4)
        flow_coarse = torch.zeros(B, 2, Hc, Wc, device=img1.device)
        h = None

        for _ in range(self.n_iters):
            
            # Warp f2 by current flow estimate at coarse scale --- defined below
            f2_warped = warp_features(f2, flow_coarse)

            # Recompute cost volume from updated f2
            cv    = self.cost_vol(f1, f2_warped)
            cv_f  = self.cv_enc(cv)

            # Encode current flow
            flow_f = self.flow_enc(flow_coarse)

            # Fuse all context for GRU
            fused  = torch.cat([f1, cv_f, flow_f], dim=1)
            h      = self.gru(fused, h)

            # Update coarse flow residual
            flow_coarse = flow_coarse + self.flow_head_coarse(h)

        return self.decoder(h, s2, s1)


# ── Cell 34: Attention Gate (unused but preserved) ───────────

"""
Instead of blindly using skip connections, this gate learns where to focus
Data flow:
    skip, context → upsample(context)
                  → concatenate(skip, context)
                  → 1x1 Conv → Sigmoid → attention map g
                  → element-wise multiply: skip * g

Output:
    gated skip features (filtered spatially)
"""
class AttentionGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, skip, context):
        context_up = F.interpolate(context, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        g = self.gate(torch.cat([skip, context_up], dim=1))
        return skip * g
