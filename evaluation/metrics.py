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

def perturbation_survival_rate(
    x_original_mel: torch.Tensor,
    x_protected_mel: torch.Tensor,
    purifier: nn.Module,
    seed: int = 2026,
) -> float:
    """
    Perturbation Survival Rate (PSR) — the correctly-normalised survival metric.

    Definition
    ----------
        PSR = || DiffPure(x + δ) - DiffPure(x) || / || δ ||

    where δ = x_protected_mel - x_original_mel and BOTH DiffPure calls use the
    SAME noise draw (fixed ``seed``) so the numerator reflects only the response
    to δ, not DiffPure's own stochasticity.

    Interpretation
    --------------
    - PSR ≈ 1.0 : δ passes through purification intact  →  it SURVIVES.
    - PSR ≈ 0.0 : DiffPure(x+δ) ≈ DiffPure(x)  →  δ was REMOVED.
    - PSR  > 1.0 : purifier amplifies δ's direction (chaotic divergence in the
                   DDPM loop), which is NOT evidence of survival.

    Why this replaces PRR as the headline number
    ---------------------------------------------
    The legacy ``purification_resistance_rate`` below is degenerate: it equals
    exactly 1.0 when δ = 0, so a reported PRR ≈ 1.0 is indistinguishable from
    "no perturbation at all". PSR is normalised by ‖δ‖ and is 0 when δ = 0, so
    it actually measures what fraction of the perturbation survives.

    Parameters
    ----------
    x_original_mel  : [batch, mel_bins, time]
    x_protected_mel : [batch, mel_bins, time]  (must describe the SAME signal
                      whose speaker-similarity / quality you report — see the
                      "unified measured signal" discipline in evaluate.py)
    purifier        : nn.Module. If it accepts a ``seed`` kwarg (MelDiffPure),
                      both calls are made deterministic; otherwise it is called
                      without one (single-pass surrogates are deterministic).
    seed            : fixed DiffPure noise seed shared by both calls.

    Returns
    -------
    psr : float   (nan if ‖δ‖ ≈ 0)
    """
    purifier.eval()
    delta_norm = (x_protected_mel - x_original_mel).norm().item()
    if delta_norm < 1e-12:
        logger.warning("PSR undefined: perturbation is ~zero (||δ||=%.2e).", delta_norm)
        return float("nan")

    def _purify(x):
        with torch.no_grad():
            try:
                return purifier(x, seed=seed)
            except TypeError:
                return purifier(x)

    y_orig = _purify(x_original_mel)
    y_prot = _purify(x_protected_mel)

    survived = (y_prot - y_orig).norm().item()
    psr = survived / delta_norm
    logger.debug("PSR: survived=%.4f | ||δ||=%.4f | PSR=%.4f",
                 survived, delta_norm, psr)
    return psr


def purification_resistance_rate(
    x_original_mel: torch.Tensor,
    x_protected_mel: torch.Tensor,
    purifier: nn.Module,
) -> float:
    """
    LEGACY — Purification Resistance Rate (PRR).

    ⚠ Degenerate: equals exactly 1.0 when δ = 0, so a value near 1.0 is
    indistinguishable from "no perturbation". Retained only for backward
    compatibility with older result CSVs. Use ``perturbation_survival_rate``
    (PSR) for any reported survival claim.

    Definition
    ----------
    PRR = ||purifier(x_protected) - x_protected||
           / ||purifier(x_original) - x_original||

    Parameters
    ----------
    x_original_mel  : torch.Tensor  [batch, mel_bins, time]
    x_protected_mel : torch.Tensor  [batch, mel_bins, time]
    purifier        : nn.Module     Surrogate purifier.

    Returns
    -------
    prr : float
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
        'psr', 'prr', 'speaker_cosine_sim', 'mel_distance', 'pesq', 'stoi', 'snr_db'

    IMPORTANT — unified measured signal
    -----------------------------------
    ``mel_protected`` MUST be the log-mel of ``wav_protected`` (the audio you
    actually ship), so that every metric — speaker similarity, survival, and
    quality — describes ONE signal. Passing a separately-null-projected mel here
    (a different perturbation than the delivered waveform) is the mismatch that
    finding #1 of the review flagged.
    """
    metrics = {}

    metrics["psr"] = perturbation_survival_rate(
        mel_original, mel_protected, purifier
    )
    metrics["prr"] = purification_resistance_rate(  # legacy, degenerate at δ=0
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
