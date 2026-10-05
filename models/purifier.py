"""
models/purifier.py
==================
Surrogate purifier model — a stand-in for DiffPure.

Why a surrogate?
----------------
Full DiffPure requires large GPU memory and hours of diffusion sampling.
For our null-space perturbation research we need a *differentiable*,
*fast*, and *locally faithful* approximation.

Architecture: UNet-style mel-spectrogram denoiser
--------------------------------------------------
Input/Output: log-mel spectrogram tensor of shape [batch, 80, time].

The network uses 1-D convolutions along the time axis with skip connections,
mimicking the structure of many speech enhancement models.  The key
properties needed for our method are:

  1. Fully differentiable (no stochastic sampling in forward pass).
  2. Produces a smooth, stable Jacobian that can be estimated with K=50 probes.
  3. Trained-free / randomly-initialised by default (good for unit tests).
     Optionally loads a pretrained speech enhancement checkpoint.

Pretrained option
-----------------
We expose `load_pretrained()` which loads torchaudio's SQUIM_OBJECTIVE
or, if unavailable, falls back to a randomly-initialised UNet.
SQUIM runs on mel spectrograms and is differentiable.

Usage
-----
    purifier = MelUNetPurifier()
    x_mel = torch.randn(1, 80, 128)   # [batch, mel_bins, time]
    x_clean = purifier(x_mel)          # same shape
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Basic residual conv block:  Conv1d → GroupNorm → GELU → Conv1d → skip.
    Operates on [batch, channels, time].
    """

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class DownBlock(nn.Module):
    """Downsample in time by 2 with a strided convolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    """Upsample in time by 2 with ConvTranspose1d, then merge skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential off-by-one in time dimension after upsample
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        return F.gelu(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# UNet Purifier
# ---------------------------------------------------------------------------

class MelUNetPurifier(nn.Module):
    """
    Lightweight UNet denoiser operating in log-mel spectrogram space.

    Architecture summary (channels along time axis):
        Encoder: 80 → 64 → 128 → 256
        Bottleneck: 256 → 256 (two ResBlocks)
        Decoder: 256+256 → 128 → 64+128 → 64 → 80

    Input:  [batch, n_mels=80, time]
    Output: [batch, n_mels=80, time]  (same shape, denoised)
    """

    def __init__(self, n_mels: int = 80, base_channels: int = 64, residual: bool = False):
        super().__init__()
        C = base_channels  # 64
        # residual=True: output = x + UNet(x)  (exact identity at zero noise;
        # used by the v2 multi-speaker DiffPure backbone)
        self.residual = residual

        # Encoder
        self.enc_in = nn.Conv1d(n_mels, C, kernel_size=3, padding=1)
        self.enc_block1 = ConvBlock(C)
        self.down1 = DownBlock(C, C * 2)       # C → 2C, time/2

        self.enc_block2 = ConvBlock(C * 2)
        self.down2 = DownBlock(C * 2, C * 4)   # 2C → 4C, time/4

        # Bottleneck
        self.bottleneck1 = ConvBlock(C * 4)
        self.bottleneck2 = ConvBlock(C * 4)

        # Decoder
        self.up1 = UpBlock(C * 4, C * 2, C * 2)   # time/4 → time/2
        self.dec_block1 = ConvBlock(C * 2)

        self.up2 = UpBlock(C * 2, C, C)            # time/2 → time
        self.dec_block2 = ConvBlock(C)

        self.out_conv = nn.Conv1d(C, n_mels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [batch, n_mels, time]
            Log-mel spectrogram.

        Returns
        -------
        x_clean : torch.Tensor, shape [batch, n_mels, time]
            Purified log-mel spectrogram.
        """
        # Ensure 3D: [batch, mel, time]
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        # Encoder
        e0 = F.gelu(self.enc_in(x))             # [B, C, T]
        e1 = self.enc_block1(e0)                 # [B, C, T]
        e2 = self.down1(e1)                      # [B, 2C, T/2]
        e2 = self.enc_block2(e2)                 # [B, 2C, T/2]
        e3 = self.down2(e2)                      # [B, 4C, T/4]

        # Bottleneck
        b = self.bottleneck1(e3)
        b = self.bottleneck2(b)                  # [B, 4C, T/4]

        # Decoder
        d1 = self.up1(b, e2)                     # [B, 2C, T/2]
        d1 = self.dec_block1(d1)
        d2 = self.up2(d1, e1)                    # [B, C, T]
        d2 = self.dec_block2(d2)

        out = self.out_conv(d2)                  # [B, n_mels, T]
        if self.residual:
            out = out + x

        if squeeze:
            out = out.squeeze(0)

        return out


# ---------------------------------------------------------------------------
# Pretrained loader
# ---------------------------------------------------------------------------

def load_pretrained(
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> "MelUNetPurifier":
    """
    Return a MelUNetPurifier.

    Priority
    --------
    1. If ``checkpoint_path`` is given and exists → load trained weights.
    2. If ``checkpoints/purifier_trained.pt`` exists next to this file's
       package root → load trained weights automatically.
    3. Fallback: randomly-initialised UNet (valid for unit tests and initial
       null-space extraction; Jacobian structure is non-trivial but random).

    Training the checkpoint
    -----------------------
    Run:
        python main.py --mode train --input data/samples/ --epochs 300

    This trains on noisy/clean mel pairs that simulate DiffPure's forward
    diffusion, giving a Jacobian that reflects a real denoiser's behaviour.

    Parameters
    ----------
    device : torch.device, optional
        Target device.  Defaults to CUDA if available, else CPU.
    checkpoint_path : str, optional
        Explicit path to a .pt checkpoint file.  Auto-discovered if None.

    Returns
    -------
    model : MelUNetPurifier
        Differentiable surrogate purifier, set to eval mode.
    """
    import os

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            logger.info("CUDA not available — running on CPU.")

    # --- Locate checkpoint ------------------------------------------------
    if checkpoint_path is None:
        # Auto-discover: <project_root>/checkpoints/purifier_trained.pt
        _this_dir    = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_this_dir)
        checkpoint_path = os.path.join(_project_root, "checkpoints", "purifier_trained.pt")

    model = MelUNetPurifier(n_mels=80, base_channels=64)

    if os.path.isfile(checkpoint_path):
        logger.info("Loading trained purifier checkpoint: %s", checkpoint_path)
        print(f"[Purifier] Loading trained checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Support both bare state_dict and wrapped dict from train_purifier.py
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        best_loss = ckpt.get("best_loss", "N/A")
        trained_epochs = ckpt.get("epochs", "?")
        print(f"[Purifier] Trained checkpoint loaded | "
              f"epochs={trained_epochs} | best_loss={best_loss}")
    else:
        logger.warning(
            "No trained checkpoint found at '%s'. "
            "Using randomly-initialised purifier. "
            "Run: python main.py --mode train --input data/samples/ --epochs 300",
            checkpoint_path,
        )
        print(f"[Purifier] WARNING: No checkpoint at {checkpoint_path}")
        print("[Purifier] Using randomly-initialised purifier.")
        print("[Purifier] Tip: run  python main.py --mode train  to train it first.")

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "MelUNetPurifier loaded on %s | parameters = %d (%.2f M)",
        device, n_params, n_params / 1e6,
    )
    print(f"[Purifier] MelUNetPurifier ready on {device} | {n_params/1e6:.2f}M params")

    return model


# ---------------------------------------------------------------------------
# 2-D spectrogram UNet (mel x time), residual — used when the 1-D UNet cannot
# learn a speech prior strong enough to denoise (v2: denoising gain 6-14 %).
# Same I/O contract as MelUNetPurifier: [B, n_mels, T] -> [B, n_mels, T].
# ---------------------------------------------------------------------------
class _Conv2Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(h + self.skip(x))


class MelUNet2D(nn.Module):
    """Residual 2-D UNet: out = x + f(x). Downsamples mel and time by 2 twice."""

    def __init__(self, n_mels: int = 80, base_channels: int = 32, residual: bool = True):
        super().__init__()
        C = base_channels
        self.residual = residual
        self.n_mels = n_mels
        self.e1 = _Conv2Block(1, C)
        self.e2 = _Conv2Block(C, 2 * C)
        self.e3 = _Conv2Block(2 * C, 4 * C)
        self.b = _Conv2Block(4 * C, 4 * C)
        self.up2 = nn.ConvTranspose2d(4 * C, 2 * C, 2, stride=2)
        self.d2 = _Conv2Block(4 * C, 2 * C)
        self.up1 = nn.ConvTranspose2d(2 * C, C, 2, stride=2)
        self.d1 = _Conv2Block(2 * C, C)
        self.out = nn.Conv2d(C, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start at identity

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        B, M, T = x.shape
        pm = (-M) % 4; pt = (-T) % 4
        h = F.pad(x, (0, pt, 0, pm)).unsqueeze(1)          # [B,1,M',T']
        e1 = self.e1(h)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        b = self.b(e3)
        d2 = self.d2(torch.cat([self.up2(b), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        o = self.out(d1).squeeze(1)[:, :M, :T]
        o = o + x if self.residual else o
        return o.squeeze(0) if squeeze else o
