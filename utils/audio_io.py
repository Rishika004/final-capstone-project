"""
utils/audio_io.py
=================
Audio I/O utilities that work without FFmpeg on Windows.

torchaudio 2.9.x requires torchcodec (which needs FFmpeg) for load/save.
This module bypasses that entirely using soundfile, which reads/writes WAV
files natively on all platforms with no extra dependencies.

All other torchaudio operations (MelSpectrogram, Resample, etc.) are
pure PyTorch and work fine — only the file I/O is replaced here.
"""

import os
import logging
from typing import Tuple

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


def load_wav(
    path: str,
    target_sr: int = 16000,
    max_seconds: float = 10.0,
) -> Tuple[torch.Tensor, int]:
    """
    Load a WAV file and return (waveform, sample_rate).

    Parameters
    ----------
    path : str
    target_sr : int
        If the file's SR differs, resample to this rate using torchaudio.transforms.
    max_seconds : float
        Clip to this many seconds to avoid OOM.

    Returns
    -------
    wav : torch.Tensor, shape [1, T]  (mono, float32, range [-1, 1])
    sr  : int  (always target_sr after resampling)
    """
    data, sr = sf.read(path, dtype='float32', always_2d=True)  # [T, C]
    wav = torch.from_numpy(data.T)  # [C, T]

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        logger.debug("Resampled %s: %d → %d Hz", os.path.basename(path), sr, target_sr)
        sr = target_sr

    # Clip to max_seconds
    max_samples = int(max_seconds * sr)
    if wav.shape[-1] > max_samples:
        wav = wav[:, :max_samples]

    return wav.float(), sr


def save_wav(
    wav: torch.Tensor,
    path: str,
    sample_rate: int = 16000,
) -> None:
    """
    Save a waveform tensor to a WAV file using soundfile.

    Parameters
    ----------
    wav : torch.Tensor, shape [1, T] or [T]
    path : str
    sample_rate : int
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = wav.detach().cpu().float().clamp(-1.0, 1.0)
    if data.dim() == 2:
        data = data.squeeze(0)  # [T]
    sf.write(path, data.numpy(), sample_rate)
    logger.debug("Saved audio: %s", path)
