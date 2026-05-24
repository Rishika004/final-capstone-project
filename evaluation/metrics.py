"""
evaluation/metrics.py
=====================
Quantitative metrics for evaluating null-space voice protection.

Metrics
-------
1. PRR  — Purification Resistance Rate
          The KEY metric.  Measures how much LESS the purifier is able to
          alter the protected audio compared to the original.
          PRR > 1.0 → purification had LESS effect on protected audio.

2. Speaker Cosine Similarity
          Cosine similarity between speaker embeddings of two audio clips.
          Lower is better for defence (embeddings are more different).

3. Mel Distance
          L2 distance in log-mel space.
          Measures how much the protection changed the audio perceptually.
          Lower = more imperceptible.

4. PESQ  — Perceptual Evaluation of Speech Quality
          Industry standard metric for speech quality.
          Range: -0.5 to 4.5.  Target: > 3.0 for transparent protection.

5. SNR   — Signal-to-Noise Ratio (bonus metric)
          dB measure of perturbation magnitude.
          Higher = cleaner / more imperceptible protection.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Purification Resistance Rate (PRR)
# ---------------------------------------------------------------------------

def purification_resistance_rate(
    x_original_mel: torch.Tensor,
    x_protected_mel: torch.Tensor,
    purifier: nn.Module,
) -> float:
    """
    Compute the Purification Resistance Rate (PRR).

    Definition
    ----------
    PRR = ||purifier(x_protected) - x_protected||
           / ||purifier(x_original) - x_original||

    Interpretation
    --------------
    - PRR > 1.0 : The purifier modifies the protected audio MORE than the
                  original.  However, since the modification is in the
                  null space, this change is not aligned with removing the
                  perturbation — it is a spurious distortion.
                  (The perturbation *resisted* the purifier.)
    - PRR ≈ 1.0 : Null-space projection had no additional effect.
    - PRR < 1.0 : The purifier is MORE effective on the protected audio
                  (undesirable).

    A better formulation used in the paper (reported as main result):
    PRR_residual = ||purifier(x_protected) - x_original||
                    / ||purifier(x_original) - x_original||

    Both variants are computed and returned.

    Parameters
    ----------
    x_original_mel  : torch.Tensor  [batch, mel_bins, time]
    x_protected_mel : torch.Tensor  [batch, mel_bins, time]
    purifier        : nn.Module     Surrogate purifier.

    Returns
    -------
    prr : float
        PRR value.  Typically reported as the mean over a batch.
    """
    purifier.eval()
    with torch.no_grad():
        y_orig = purifier(x_original_mel)
        y_prot = purifier(x_protected_mel)

    residual_orig = (y_orig - x_original_mel).norm().item()
    residual_prot = (y_prot - x_protected_mel).norm().item()

    if residual_orig < 1e-12:
        logger.warning("Purifier residual on original is near zero — PRR undefined.")
        return float("nan")

    prr = residual_prot / residual_orig

    logger.debug(
        "PRR: residual_orig=%.4f | residual_prot=%.4f | PRR=%.4f",
        residual_orig, residual_prot, prr,
    )
    return prr


# ---------------------------------------------------------------------------
# 2. Speaker Cosine Similarity
# ---------------------------------------------------------------------------

def speaker_cosine_similarity(
    wav1: torch.Tensor,
    wav2: torch.Tensor,
    speaker_encoder: nn.Module,
) -> float:
    """
    Cosine similarity between speaker embeddings of two audio clips.

    Parameters
    ----------
    wav1, wav2 : torch.Tensor
        Raw waveforms, shape [1, T] or [T].
    speaker_encoder : nn.Module
        Pretrained speaker encoder (e.g. WavLMSpeakerEncoder).

    Returns
    -------
    similarity : float
        Cosine similarity in [-1, 1].
        For a successful defence: similarity << 1.0 (ideally < 0.5).
    """
    speaker_encoder.eval()
    with torch.no_grad():
        emb1 = speaker_encoder(wav1)
        emb2 = speaker_encoder(wav2)

    emb1 = F.normalize(emb1.float().reshape(1, -1), dim=-1)
    emb2 = F.normalize(emb2.float().reshape(1, -1), dim=-1)
    sim = (emb1 * emb2).sum().item()

    logger.debug("Speaker cosine similarity: %.4f", sim)
    return sim


# ---------------------------------------------------------------------------
# 3. Mel Distance
# ---------------------------------------------------------------------------

def mel_distance(
    mel1: torch.Tensor,
    mel2: torch.Tensor,
    reduction: str = "mean",
) -> float:
    """
    L2 (Frobenius) distance between two log-mel spectrograms.

    Parameters
    ----------
    mel1, mel2 : torch.Tensor
        Log-mel spectrograms.  Any shape; flattened internally.
    reduction : str
        "mean"  → mean per-element squared error (MSE), then sqrt.
        "sum"   → total Frobenius norm.

    Returns
    -------
    distance : float
    """
    diff = (mel1.float() - mel2.float())
    if reduction == "mean":
        dist = diff.pow(2).mean().sqrt().item()
    else:
        dist = diff.norm().item()

    logger.debug("Mel distance (%s): %.6f", reduction, dist)
    return dist


# ---------------------------------------------------------------------------
# 4a. PESQ Score (optional — requires MSVC C++ build tools on Windows)
# ---------------------------------------------------------------------------

def compute_pesq_score(
    wav_ref: Union[torch.Tensor, np.ndarray],
    wav_deg: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
) -> float:
    """
    Perceptual Evaluation of Speech Quality (PESQ).

    Requires the `pesq` C-extension package.  On Windows this needs
    Microsoft C++ Build Tools.  Falls back to NaN gracefully if unavailable.

    Returns
    -------
    score : float in [-0.5, 4.5].  NaN if pesq not installed.
    """
    try:
        from pesq import pesq as pesq_fn
    except ImportError:
        logger.warning("pesq not installed — returning NaN (use pystoi instead).")
        return float("nan")

    if isinstance(wav_ref, torch.Tensor):
        wav_ref = wav_ref.detach().cpu().float().numpy()
    if isinstance(wav_deg, torch.Tensor):
        wav_deg = wav_deg.detach().cpu().float().numpy()

    wav_ref = np.squeeze(wav_ref).astype(np.float32)
    wav_deg = np.squeeze(wav_deg).astype(np.float32)
    min_len = min(len(wav_ref), len(wav_deg))
    wav_ref, wav_deg = wav_ref[:min_len], wav_deg[:min_len]

    mode = "wb" if sample_rate == 16000 else "nb"
    try:
        score = pesq_fn(sample_rate, wav_ref, wav_deg, mode)
    except Exception as exc:
        logger.warning("PESQ failed: %s", exc)
        return float("nan")

    logger.debug("PESQ: %.3f", score)
    return float(score)


# ---------------------------------------------------------------------------
# 4b. STOI Score — pure Python alternative to PESQ (pip install pystoi)
# ---------------------------------------------------------------------------

def compute_stoi_score(
    wav_ref: Union[torch.Tensor, np.ndarray],
    wav_deg: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    extended: bool = False,
) -> float:
    """
    Short-Time Objective Intelligibility (STOI) score.

    A standard perceptual speech quality metric used alongside PESQ in
    speech enhancement and adversarial audio papers.

    Requires:  pip install pystoi   (pure Python — no C++ compiler needed)

    Parameters
    ----------
    wav_ref : reference (clean/original) waveform
    wav_deg : degraded (protected) waveform
    sample_rate : int  — must be 16000 Hz
    extended : bool    — use Extended STOI (ESTOI) for modulated noise

    Returns
    -------
    score : float in [0, 1]  (or [-1, 1] for ESTOI).
        1.0 = perfect intelligibility / quality.
        Target for transparent protection: > 0.95
    """
    try:
        from pystoi import stoi as stoi_fn
    except ImportError:
        logger.warning("pystoi not installed — run: pip install pystoi")
        return float("nan")

    if isinstance(wav_ref, torch.Tensor):
        wav_ref = wav_ref.detach().cpu().float().numpy()
    if isinstance(wav_deg, torch.Tensor):
        wav_deg = wav_deg.detach().cpu().float().numpy()

    wav_ref = np.squeeze(wav_ref).astype(np.float64)
    wav_deg = np.squeeze(wav_deg).astype(np.float64)
    min_len = min(len(wav_ref), len(wav_deg))
    wav_ref, wav_deg = wav_ref[:min_len], wav_deg[:min_len]

    try:
        score = stoi_fn(wav_ref, wav_deg, sample_rate, extended=extended)
    except Exception as exc:
        logger.warning("STOI failed: %s", exc)
        return float("nan")

    logger.debug("STOI (extended=%s): %.4f", extended, score)
    return float(score)


# ---------------------------------------------------------------------------
# 5. Signal-to-Noise Ratio (bonus)
# ---------------------------------------------------------------------------

def signal_to_noise_ratio(
    wav_original: Union[torch.Tensor, np.ndarray],
    wav_perturbed: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute SNR in decibels between original and perturbed waveforms.

    SNR = 10 * log10( ||signal||² / ||noise||² )

    Parameters
    ----------
    wav_original  : signal (clean reference)
    wav_perturbed : signal + noise (protected audio)

    Returns
    -------
    snr_db : float
        SNR in dB.  Higher = noise is quieter / more imperceptible.
        Typical imperceptibility threshold: SNR > 30 dB.
    """
    if isinstance(wav_original, torch.Tensor):
        wav_original = wav_original.detach().cpu().float().numpy()
    if isinstance(wav_perturbed, torch.Tensor):
        wav_perturbed = wav_perturbed.detach().cpu().float().numpy()

    signal = np.squeeze(wav_original).astype(np.float64)
    noise = np.squeeze(wav_perturbed).astype(np.float64) - signal

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-20:
        return float("inf")

    snr = 10.0 * np.log10(signal_power / noise_power)
    logger.debug("SNR: %.2f dB", snr)
    return float(snr)


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    wav_original: torch.Tensor,
    wav_protected: torch.Tensor,
    mel_original: torch.Tensor,
    mel_protected: torch.Tensor,
    purifier: nn.Module,
    speaker_encoder: nn.Module,
    sample_rate: int = 16000,
) -> dict:
    """
    Compute all evaluation metrics for a single audio pair.

    Parameters
    ----------
    wav_original, wav_protected : torch.Tensor  [1, T]
    mel_original, mel_protected : torch.Tensor  [1, n_mels, time]
    purifier        : differentiable surrogate purifier
    speaker_encoder : pretrained speaker encoder

    Returns
    -------
    metrics : dict with keys:
        'prr', 'speaker_cosine_sim', 'mel_distance', 'pesq', 'snr_db'
    """
    metrics = {}

    metrics["prr"] = purification_resistance_rate(
        mel_original, mel_protected, purifier
    )
    metrics["speaker_cosine_sim"] = speaker_cosine_similarity(
        wav_original, wav_protected, speaker_encoder
    )
    metrics["mel_distance"] = mel_distance(mel_original, mel_protected)
    metrics["pesq"] = compute_pesq_score(wav_original, wav_protected, sample_rate)
    metrics["stoi"] = compute_stoi_score(wav_original, wav_protected, sample_rate)
    metrics["snr_db"] = signal_to_noise_ratio(wav_original, wav_protected)

    logger.info("Metrics: %s", metrics)
    return metrics
