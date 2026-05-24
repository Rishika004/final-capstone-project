"""
training/train_purifier.py
==========================
Train the surrogate purifier on noisy/clean mel-spectrogram pairs so it
actually approximates DiffPure's local linear denoising behaviour.

Why this matters
----------------
Without training, the purifier is randomly initialised.  Its Jacobian is
random noise — the null space is arbitrary and provides no real guarantee
that the perturbation will survive the *real* DiffPure operator.

After training on noisy→clean mel pairs, the Jacobian reflects the actual
local linear mapping of a denoiser, and null(J) is the subspace that the
denoiser truly ignores.  Perturbations projected into this null space
will survive DiffPure-style purification.

DiffPure simulation
-------------------
DiffPure:
  1. Adds Gaussian noise at scale sigma_t (forward diffusion step).
  2. Runs a denoising score-matching model to remove it (reverse step).

We simulate this with a UNet denoiser trained on:
  x_noisy = x_clean + N(0, sigma)   for sigma ∈ {0.05, 0.1, 0.2, 0.3}

The trained UNet captures the first-order (Jacobian) behaviour of DiffPure
around the data manifold.

Usage
-----
    # Standalone
    python training/train_purifier.py --data data/samples/ --epochs 300

    # Via main CLI
    python main.py --mode train --input data/samples/ --epochs 300
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T

# Allow running as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.purifier import MelUNetPurifier
from utils.audio_io import load_wav

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match evaluate.py)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
N_MELS      = 80
N_FFT       = 1024
HOP_LENGTH  = 256
MAX_SEC     = 10.0

CHECKPOINT_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"
)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "purifier_trained.pt")

# Noise levels that simulate DiffPure's forward-diffusion sigma range
DEFAULT_NOISE_SIGMAS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)


# ---------------------------------------------------------------------------
# Mel helper
# ---------------------------------------------------------------------------

_mel_fn_cache: dict = {}   # keyed by str(device) to handle CPU/CUDA switches


def _get_mel_fn(device: torch.device) -> T.MelSpectrogram:
    key = str(device)
    if key not in _mel_fn_cache:
        _mel_fn_cache[key] = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        ).to(device)
    return _mel_fn_cache[key]


def wav_to_logmel(wav: torch.Tensor, device: torch.device) -> torch.Tensor:
    """[1, T] → [1, N_MELS, frames]"""
    fn = _get_mel_fn(device)
    mel = fn(wav.to(device))
    return torch.log(mel + 1e-9)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

CROP_FRAMES = 188   # fixed crop length = ~3s at 16kHz/hop256 (80 × 188 mel)


def build_dataset(
    audio_dir: str,
    device: torch.device,
    noise_sigmas: Tuple[float, ...] = DEFAULT_NOISE_SIGMAS,
    repeats_per_sigma: int = 3,
    crop_frames: int = CROP_FRAMES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (x_noisy, x_clean) mel-spectrogram pairs using RANDOM CROPS.

    Each clip is cropped to a fixed length (crop_frames = ~3s) at a random
    offset so that:
      - All samples have the same shape → can be stacked into a batch.
      - Long clips contribute multiple different crops (via repeats).
      - Short clips (< crop_frames) are zero-padded on the right.
      - No audio is wasted by truncating everything to the shortest clip.

    With 500 LJSpeech files × 5 sigmas × 3 repeats = 7,500 training samples.

    Returns
    -------
    X_noisy : FloatTensor [N, N_MELS, crop_frames]
    X_clean : FloatTensor [N, N_MELS, crop_frames]
    """
    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in '{audio_dir}'")

    print(f"[Train] Found {len(wav_files)} audio file(s)")

    # Load all mels (variable length)
    mels = []
    skipped = 0
    for wav_path in wav_files:
        try:
            wav, _ = load_wav(str(wav_path), target_sr=SAMPLE_RATE, max_seconds=MAX_SEC)
            mel = wav_to_logmel(wav, device).squeeze(0)   # [N_MELS, T]
            mels.append(mel)
        except Exception as exc:
            logger.warning("Skipping %s: %s", wav_path.name, exc)
            skipped += 1

    if not mels:
        raise RuntimeError("Could not load any audio files.")

    if skipped:
        print(f"[Train] Skipped {skipped} files due to load errors.")

    t_values = [m.shape[-1] for m in mels]
    print(f"[Train] Loaded {len(mels)} files | "
          f"frame range: {min(t_values)}–{max(t_values)} | "
          f"crop: {crop_frames} frames (~{crop_frames*256/16000:.1f}s)")

    X_noisy_list, X_clean_list = [], []
    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    cpu_rng = torch.Generator()   # for crop offsets (CPU)
    cpu_rng.manual_seed(42)

    for mel_clean in mels:
        T = mel_clean.shape[-1]

        # Pad if shorter than crop window
        if T < crop_frames:
            pad = crop_frames - T
            mel_clean = torch.nn.functional.pad(mel_clean, (0, pad))
            T = crop_frames

        for sigma in noise_sigmas:
            for _ in range(repeats_per_sigma):
                # Random crop offset
                max_offset = T - crop_frames
                offset = int(torch.randint(0, max_offset + 1, (1,),
                                           generator=cpu_rng).item())
                crop = mel_clean[:, offset : offset + crop_frames]  # [N_MELS, crop_frames]

                noise     = torch.randn(crop.shape, generator=rng,
                                        device=device) * sigma
                mel_noisy = crop + noise
                X_noisy_list.append(mel_noisy)
                X_clean_list.append(crop)

    X_noisy = torch.stack(X_noisy_list)   # [N, N_MELS, crop_frames]
    X_clean = torch.stack(X_clean_list)

    print(f"[Train] Dataset: {X_noisy.shape[0]} samples | "
          f"shape [{N_MELS}, {crop_frames}] | "
          f"sigma range [{min(noise_sigmas)}, {max(noise_sigmas)}]")
    return X_noisy, X_clean


# ---------------------------------------------------------------------------
# Loss: L1 + spectral convergence (better for spectrograms than plain MSE)
# ---------------------------------------------------------------------------

def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Spectral convergence loss — penalises the relative Frobenius error in
    linear (non-log) mel space, which better matches perceptual distance.
    """
    pred_lin   = torch.exp(pred)
    target_lin = torch.exp(target)
    sc = torch.norm(target_lin - pred_lin, p="fro") / (torch.norm(target_lin, p="fro") + 1e-8)
    return sc


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """alpha * L1  +  (1-alpha) * spectral convergence"""
    l1  = nn.functional.l1_loss(pred, target)
    sc  = spectral_loss(pred, target)
    return alpha * l1 + (1.0 - alpha) * sc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_purifier(
    audio_dir: str,
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-3,
    noise_sigmas: Tuple[float, ...] = DEFAULT_NOISE_SIGMAS,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = CHECKPOINT_PATH,
    log_interval: int = 25,
) -> MelUNetPurifier:
    """
    Train MelUNetPurifier on noisy→clean mel pairs.

    Algorithm
    ---------
    For t in 1..epochs:
        for each mini-batch (x_noisy, x_clean):
            loss = alpha * L1(model(x_noisy), x_clean)
                 + (1-alpha) * SpectralConvergence(model(x_noisy), x_clean)
            loss.backward()  →  Adam step  →  cosine LR decay

    Saves checkpoint to checkpoints/purifier_trained.pt.

    Returns
    -------
    model : MelUNetPurifier  (trained, eval mode)
    """
    print("=" * 60)
    print("STEP 4 — TRAINING SURROGATE PURIFIER")
    print(f"  audio_dir    = {audio_dir}")
    print(f"  epochs       = {epochs}")
    print(f"  batch_size   = {batch_size}")
    print(f"  lr           = {lr}")
    print(f"  noise_sigmas = {noise_sigmas}")
    print(f"  device       = {device}")
    print(f"  checkpoint   = {checkpoint_path}")
    print("=" * 60)

    # ---- 1. Build dataset ------------------------------------------------
    print("\n[Train] Building noisy/clean mel dataset ...")
    X_noisy, X_clean = build_dataset(audio_dir, device, noise_sigmas)
    N = X_noisy.shape[0]

    # ---- 2. Model, optimizer, scheduler ----------------------------------
    model     = MelUNetPurifier(n_mels=N_MELS, base_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model: {n_params/1e6:.2f}M parameters")
    print(f"[Train] Training on {N} samples for {epochs} epochs ...")

    # ---- 3. Training loop ------------------------------------------------
    best_loss  = float("inf")
    best_state = None
    t0         = time.perf_counter()

    model.train()
    for epoch in range(1, epochs + 1):
        perm       = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, N, batch_size):
            idx    = perm[i : i + batch_size]
            xn     = X_noisy[idx]
            xc     = X_clean[idx]

            optimizer.zero_grad()
            pred   = model(xn)
            loss   = combined_loss(pred, xc)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.perf_counter() - t0
            lr_now  = scheduler.get_last_lr()[0]
            print(
                f"[Train] epoch {epoch:4d}/{epochs} | "
                f"loss={avg_loss:.5f} | best={best_loss:.5f} | "
                f"lr={lr_now:.2e} | t={elapsed:.1f}s"
            )

    # ---- 4. Load best weights & save checkpoint --------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_loss":        best_loss,
            "epochs":           epochs,
            "noise_sigmas":     noise_sigmas,
            "n_mels":           N_MELS,
            "base_channels":    64,
        },
        checkpoint_path,
    )
    print(f"\n[Train] Checkpoint saved to: {checkpoint_path}")
    print(f"[Train] Best training loss : {best_loss:.5f}")
    print("[Train] Done — purifier ready for null-space extraction.\n")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train surrogate DiffPure purifier")
    parser.add_argument("--data",       default="data/samples/",
                        help="Directory of training .wav files")
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--batch-size", type=int,   default=16, dest="batch_size")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    dev = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else torch.device(args.device))

    train_purifier(
        audio_dir  = args.data,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        device     = dev,
    )
