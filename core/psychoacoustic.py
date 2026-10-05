"""
core/psychoacoustic.py
======================
Frequency-masking model for perceptually-shaped perturbations.

Why this exists
---------------
The original PGD pipeline used a flat L-inf waveform budget (epsilon_wav = 0.05,
signed-gradient steps). Signed gradients saturate the budget almost everywhere,
so the delta is effectively +-0.05 broadband noise: audible hiss across the whole
file, loudest (relatively) in pauses and quiet phonemes, with ~50% of its energy
above 4 kHz where the ear is most sensitive to noise. That is why the protected
audio "does not sound like the original".

The fix is standard psychoacoustics (Johnston 1988; used for adversarial audio
by Qin et al., ICML 2019, and by AntiFake, CCS 2023): a listener cannot hear
noise whose short-time spectrum stays below the *masking threshold* of the
speech itself. Loud spectro-temporal regions of the voice "mask" nearby
frequencies; silence masks nothing (threshold falls to the absolute threshold
of hearing). If every STFT bin of the perturbation is kept under this
threshold, the perturbation is inaudible BY CONSTRUCTION — and the budget
automatically concentrates where the voice is loud, i.e. exactly where speaker
encoders look.

This module provides
--------------------
1. ``PsychoacousticMasker`` — computes a per-frame, per-bin linear magnitude
   cap ``C[f, t]`` from the ORIGINAL waveform (simplified Johnston model:
   Bark-domain two-slope spreading + tonality offset + absolute threshold of
   hearing floor).
2. ``project_delta_to_masking`` — the "perceptual projection": clamps the STFT
   magnitude of a waveform delta to ``C[f, t]`` (optionally band-limited) and
   resynthesises. Used inside PGD exactly where the old code used
   ``clamp_(-eps, +eps)``.
3. ``masking_penalty`` — a differentiable hinge loss on threshold violations,
   optionally added to the attack loss so the optimiser prefers in-budget
   directions instead of fighting the projection.

All computation is torch, runs on GPU, and uses the same STFT parameters as the
rest of the project (n_fft=1024, hop=256 at 16 kHz).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Bark scale + absolute threshold of hearing
# ---------------------------------------------------------------------------


def _hz_to_bark(freq_hz: torch.Tensor) -> torch.Tensor:
    """Traunmüller/Zwicker Bark scale (as used in Qin et al. 2019)."""
    f = freq_hz
    return 13.0 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)


def _ath_db(freq_hz: torch.Tensor) -> torch.Tensor:
    """
    Absolute Threshold of Hearing in dB SPL (Terhardt approximation).

    Values are on the conventional scale where the loudest representable
    signal sits at ~96 dB SPL. Below ~20 Hz the formula diverges; we clamp.
    """
    f = (freq_hz / 1000.0).clamp(min=0.02)  # kHz, avoid f=0 blow-up
    ath = (
        3.64 * f.pow(-0.8)
        - 6.5 * torch.exp(-0.6 * (f - 3.3) ** 2)
        + 1e-3 * f.pow(4)
    )
    return ath.clamp(max=96.0)


# ---------------------------------------------------------------------------
# Masker
# ---------------------------------------------------------------------------


class PsychoacousticMasker:
    """
    Computes per-frame masking-threshold magnitude caps for a waveform.

    Parameters
    ----------
    sample_rate, n_fft, hop_length : STFT parameters (must match the delta's).
    margin_db : float
        Global offset applied to the threshold. 0 = strictly at the masking
        threshold (inaudible). Positive values allow a louder perturbation
        (each +6 dB doubles the allowed amplitude) trading a little audibility
        for stronger protection. Negative = extra conservative.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        margin_db: float = 0.0,
        device: Optional[torch.device] = None,
        local_snr_floor_db: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.margin_db = margin_db
        # Per-frame energy gate. The Bark masking model floors at the absolute
        # threshold of hearing (+margin), so in pauses the cap is LOUDER than
        # the speech (measured 2026-09-11: delta 14 dB above signal in silent
        # frames -> PESQ ~1.8). With a floor of s dB, the cap is scaled per
        # frame so that the maximum delta energy is <= frame energy - s dB.
        self.local_snr_floor_db = local_snr_floor_db
        self.device = device or torch.device("cpu")

        self.window = torch.hann_window(n_fft, device=self.device)

        n_bins = n_fft // 2 + 1
        freqs = torch.linspace(0.0, sample_rate / 2.0, n_bins, device=self.device)
        bark = _hz_to_bark(freqs)                       # [F]
        self.ath_db = _ath_db(freqs)                    # [F]

        # Two-slope spreading function in the Bark domain, precomputed as a
        # power-domain matrix:  SF[i, j] = gain from masker bin j to maskee
        # bin i.  Lower slope (masking of lower frequencies): -27 dB/Bark.
        # Upper slope (masking of higher frequencies): -12 dB/Bark.
        dz = bark.unsqueeze(0) - bark.unsqueeze(1)      # dz[i, j] = z_i - z_j
        spread_db = torch.where(dz >= 0, -12.0 * dz, 27.0 * dz)
        self.spread_pow = torch.pow(10.0, spread_db / 10.0)  # [F, F]

        # Tonality offset: speech is a tonal/noise mixture; we use the
        # conservative tonal-masker offset  O(z) = 14.5 + z  (dB), which
        # under-estimates how much a noisy masker can hide. Under-estimating
        # keeps us safely inaudible.
        self.offset_db = (14.5 + bark).unsqueeze(1)     # [F, 1]

    # ------------------------------------------------------------------
    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """[1, T] or [T] waveform -> complex STFT [F, frames]."""
        w = wav.reshape(-1).to(self.device)
        return torch.stft(
            w,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Complex STFT [F, frames] -> [1, length] waveform."""
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=length,
        )
        return wav.unsqueeze(0)

    # ------------------------------------------------------------------
    def magnitude_cap(self, wav_original: torch.Tensor) -> torch.Tensor:
        """
        Per-bin masking-threshold magnitude cap for a given clean waveform.

        Returns
        -------
        cap : torch.Tensor [F, frames]
            Linear STFT-magnitude ceiling. A perturbation whose STFT magnitude
            stays below ``cap`` everywhere is (by the model) inaudible when
            played together with ``wav_original``.
        """
        with torch.no_grad():
            S = self.stft(wav_original)                     # [F, T]
            psd_db = 10.0 * torch.log10(S.abs().pow(2) + 1e-20)

            # Normalise so the file's loudest bin sits at 96 dB SPL.
            norm = 96.0 - psd_db.max()
            psd_db_n = psd_db + norm                        # [F, T]

            # Spread the (power-domain) PSD across the Bark axis.
            spread = self.spread_pow @ torch.pow(10.0, psd_db_n / 10.0)
            thr_db = 10.0 * torch.log10(spread + 1e-20) - self.offset_db

            # Floor at the absolute threshold of hearing, apply user margin.
            thr_db = torch.maximum(thr_db, self.ath_db.unsqueeze(1))
            thr_db = thr_db + self.margin_db

            # Back to the un-normalised linear magnitude scale of S.
            cap = torch.pow(10.0, (thr_db - norm) / 20.0)   # [F, T]

            # Per-frame signal energy, used by project_delta_to_masking to
            # enforce the local SNR floor on the REALISED delta (scaling the
            # cap itself is far too conservative: the cap sums to ~+29 dB
            # relative to the signal, so every bin would shrink by ~40 dB).
            self.frame_energy = S.abs().pow(2).sum(dim=0)                # [T]
        return cap


# ---------------------------------------------------------------------------
# Projection + penalty
# ---------------------------------------------------------------------------


def project_delta_to_masking(
    delta_wav: torch.Tensor,
    cap: torch.Tensor,
    masker: PsychoacousticMasker,
    bandlimit_hz: Optional[float] = None,
    hard_linf: Optional[float] = None,
) -> torch.Tensor:
    """
    Project a waveform perturbation onto the inaudibility constraint set.

    Every STFT bin of ``delta_wav`` whose magnitude exceeds the masking cap is
    scaled down to the cap (phase preserved); bins above ``bandlimit_hz`` are
    zeroed (this also restores robustness to resampling / low-pass transforms,
    which the flat perturbation lost). The result is resynthesised via iSTFT.

    Parameters
    ----------
    delta_wav : [1, T] waveform perturbation.
    cap       : [F, frames] magnitude cap from ``PsychoacousticMasker.magnitude_cap``.
    bandlimit_hz : zero all delta energy above this frequency (None = off).
    hard_linf : optional final safety clamp on the waveform samples.
    """
    length = delta_wav.shape[-1]
    D = masker.stft(delta_wav)                              # [F, T']
    Tn = min(D.shape[-1], cap.shape[-1])
    D = D[..., :Tn]
    c = cap[..., :Tn]

    mag = D.abs()
    scale = torch.where(mag > c, c / (mag + 1e-12), torch.ones_like(mag))
    D = D * scale

    if bandlimit_hz is not None:
        n_bins = masker.n_fft // 2 + 1
        cut = int(round(bandlimit_hz / (masker.sample_rate / 2.0) * (n_bins - 1)))
        D[cut:, :] = 0.0

    # Local SNR floor: per frame, delta energy <= signal energy - floor dB.
    floor = getattr(masker, "local_snr_floor_db", None)
    fe = getattr(masker, "frame_energy", None)
    if floor is not None and fe is not None:
        fe = fe[..., :Tn]
        e_d = D.abs().pow(2).sum(dim=0)                                  # [T]
        allowed = fe * 10.0 ** (-floor / 10.0)
        fscale = torch.sqrt(torch.clamp(allowed / (e_d + 1e-20), max=1.0))
        D = D * fscale.unsqueeze(0)

    out = masker.istft(D, length)
    if floor is not None and fe is not None:
        # second pass: iSTFT overlap-add leaks energy across frame boundaries
        # (loud -> silent transitions); re-analyse and gate once more.
        for _ in range(2):
            D2 = masker.stft(out)[..., :Tn]
            e_d = D2.abs().pow(2).sum(dim=0)
            fscale = torch.sqrt(torch.clamp(allowed / (e_d + 1e-20), max=1.0))
            out = masker.istft(D2 * fscale.unsqueeze(0), length)
    if hard_linf is not None:
        out = out.clamp(-hard_linf, hard_linf)
    return out


def masking_penalty(
    delta_wav: torch.Tensor,
    cap: torch.Tensor,
    masker: PsychoacousticMasker,
) -> torch.Tensor:
    """
    Differentiable hinge on masking-threshold violations (relative, in power).

    Zero when the delta is fully under the threshold; grows with the amount of
    audible (over-threshold) energy. Add ``alpha * masking_penalty(...)`` to the
    attack loss so the optimiser steers toward inaudible directions instead of
    relying purely on the post-step projection.
    """
    D = masker.stft(delta_wav)
    Tn = min(D.shape[-1], cap.shape[-1])
    over = (D.abs()[..., :Tn].pow(2) - cap[..., :Tn].pow(2)).clamp(min=0.0)
    return over.sum() / (cap[..., :Tn].pow(2).sum() + 1e-12)


# ---------------------------------------------------------------------------
# Quality metrics helper (for honest waveform-domain reporting)
# ---------------------------------------------------------------------------


def waveform_snr_db(x: torch.Tensor, y: torch.Tensor) -> float:
    """Plain waveform SNR of y vs reference x, in dB."""
    x = x.reshape(-1)
    y = y.reshape(-1)[: x.numel()]
    noise = y - x[: y.numel()]
    return 10.0 * math.log10(
        (x.pow(2).sum().item()) / (noise.pow(2).sum().item() + 1e-20)
    )


def silence_noise_db(
    x: torch.Tensor,
    y: torch.Tensor,
    sample_rate: int = 16000,
    frame_ms: float = 30.0,
    silence_rel_db: float = 40.0,
) -> Tuple[float, int]:
    """
    RMS level (dB re full scale) of the perturbation inside the ORIGINAL's
    silent frames — the single best predictor of "I can hear a hiss".

    Returns (noise_db, n_silent_frames); (nan, 0) if the file has no silence.
    """
    x = x.reshape(-1)
    y = y.reshape(-1)[: x.numel()]
    d = y - x[: y.numel()]
    fl = int(frame_ms / 1000.0 * sample_rate)
    n = (x.numel() // fl) * fl
    xe = x[:n].reshape(-1, fl).pow(2).mean(dim=1)
    de = d[:n].reshape(-1, fl).pow(2).mean(dim=1)
    thresh = xe.max() * (10.0 ** (-silence_rel_db / 10.0))
    sil = xe < thresh
    if not bool(sil.any()):
        return float("nan"), 0
    noise_db = 10.0 * math.log10(de[sil].mean().item() + 1e-20)
    return noise_db, int(sil.sum().item())
