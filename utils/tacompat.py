"""
utils/tacompat.py
=================
Pure-PyTorch replacements for the three torchaudio transforms this project
depends on. No native extension, so they keep working when Windows Smart App
Control blocks `libtorchaudio.pyd` (observed 2026-09-15: the .pyd is unsigned
and the policy is enforced, VerifiedAndReputablePolicyState = 1).

Drop-in usage, which prefers the real torchaudio whenever it loads:

    try:
        import torchaudio.transforms as T
    except Exception:
        from utils import tacompat as T

Implements `MelSpectrogram`, `InverseMelScale` and `GriffinLim` with the same
constructor arguments and the same defaults this project relies on, following
torchaudio's own formulas so existing results stay comparable:
  * STFT: hann window, center=True, pad_mode="reflect", power spectrogram
  * mel filterbank: triangular, `mel_scale` in {"htk", "slaney"},
    `norm` in {None, "slaney"}
  * InverseMelScale: least-squares solve, as torchaudio does with driver="gels"
  * GriffinLim: random phase init, momentum 0.99

`python -m utils.tacompat` self-tests the filterbank against librosa.
"""
import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mel scale conversions (identical to torchaudio.functional.functional)
# ---------------------------------------------------------------------------
def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    if mel_scale not in ("htk", "slaney"):
        raise ValueError(f'mel_scale must be "htk" or "slaney", got {mel_scale}')
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: torch.Tensor, mel_scale: str = "htk") -> torch.Tensor:
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def _triangular_filterbank(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:
    f_diff = f_pts[1:] - f_pts[:-1]                       # [n_filter + 1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # [n_freqs, n_filter + 2]
    zero = torch.zeros(1, device=all_freqs.device)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return torch.max(zero, torch.min(down_slopes, up_slopes))


def melscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_mels: int,
                    sample_rate: int, norm: Optional[str] = None,
                    mel_scale: str = "htk") -> torch.Tensor:
    """[n_freqs, n_mels] triangular mel filterbank."""
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be None or "slaney"')
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel(f_min, mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale)
    fb = _triangular_filterbank(all_freqs, f_pts)
    if norm == "slaney":
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    return fb


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class Spectrogram(nn.Module):
    def __init__(self, n_fft: int = 400, win_length: Optional[int] = None,
                 hop_length: Optional[int] = None, power: Optional[float] = 2.0,
                 center: bool = True, pad_mode: str = "reflect", **_):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 2
        self.power = power
        self.center = center
        self.pad_mode = pad_mode
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        shape = waveform.shape
        wav = waveform.reshape(-1, shape[-1])
        spec = torch.stft(wav, self.n_fft, self.hop_length, self.win_length,
                          window=self.window.to(wav.device), center=self.center,
                          pad_mode=self.pad_mode, normalized=False, onesided=True,
                          return_complex=True)
        spec = spec.reshape(shape[:-1] + spec.shape[-2:])
        if self.power is None:
            return spec
        return spec.abs().pow(self.power)


class MelScale(nn.Module):
    def __init__(self, n_mels: int = 128, sample_rate: int = 16000, f_min: float = 0.0,
                 f_max: Optional[float] = None, n_stft: int = 201,
                 norm: Optional[str] = None, mel_scale: str = "htk", **_):
        super().__init__()
        f_max = f_max if f_max is not None else float(sample_rate // 2)
        fb = melscale_fbanks(n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_scale)
        self.register_buffer("fb", fb)

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        return torch.matmul(specgram.transpose(-1, -2),
                            self.fb.to(specgram.device)).transpose(-1, -2)


class MelSpectrogram(nn.Module):
    """torchaudio.transforms.MelSpectrogram, pure torch."""

    def __init__(self, sample_rate: int = 16000, n_fft: int = 400,
                 win_length: Optional[int] = None, hop_length: Optional[int] = None,
                 f_min: float = 0.0, f_max: Optional[float] = None, n_mels: int = 128,
                 power: float = 2.0, center: bool = True, pad_mode: str = "reflect",
                 norm: Optional[str] = None, mel_scale: str = "htk", **_):
        super().__init__()
        self.spectrogram = Spectrogram(n_fft=n_fft, win_length=win_length,
                                       hop_length=hop_length, power=power,
                                       center=center, pad_mode=pad_mode)
        self.mel_scale = MelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=f_min,
                                  f_max=f_max, n_stft=n_fft // 2 + 1, norm=norm,
                                  mel_scale=mel_scale)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mel_scale(self.spectrogram(waveform))


class InverseMelScale(nn.Module):
    """Least-squares mel -> linear spectrogram, as torchaudio's driver="gels"."""

    def __init__(self, n_stft: int, n_mels: int = 128, sample_rate: int = 16000,
                 f_min: float = 0.0, f_max: Optional[float] = None,
                 norm: Optional[str] = None, mel_scale: str = "htk",
                 driver: str = "gels", **_):
        super().__init__()
        f_max = f_max if f_max is not None else float(sample_rate // 2)
        fb = melscale_fbanks(n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_scale)
        self.register_buffer("fb", fb)
        self.driver = driver

    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        shape = melspec.shape
        mel = melspec.reshape(-1, shape[-2], shape[-1])
        fb = self.fb.to(mel.device)
        sol = torch.linalg.lstsq(fb.transpose(-1, -2).expand(mel.shape[0], -1, -1),
                                 mel, driver=self.driver).solution
        return sol.clamp(min=0).reshape(shape[:-2] + sol.shape[-2:])


class GriffinLim(nn.Module):
    """torchaudio.transforms.GriffinLim (random phase init, momentum)."""

    def __init__(self, n_fft: int = 400, n_iter: int = 32,
                 win_length: Optional[int] = None, hop_length: Optional[int] = None,
                 power: float = 2.0, momentum: float = 0.99, length: Optional[int] = None,
                 rand_init: bool = True, **_):
        super().__init__()
        self.n_fft = n_fft
        self.n_iter = n_iter
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 2
        self.power = power
        self.momentum = momentum / (1 + momentum)
        self.length = length
        self.rand_init = rand_init
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        shape = specgram.shape
        spec = specgram.reshape(-1, shape[-2], shape[-1])
        spec = spec.pow(1 / self.power)
        win = self.window.to(spec.device)
        if self.rand_init:
            angles = torch.randn(spec.shape, dtype=torch.complex64, device=spec.device)
        else:
            angles = torch.ones(spec.shape, dtype=torch.complex64, device=spec.device)
        tprev = torch.tensor(0.0, device=spec.device, dtype=spec.dtype)
        for _ in range(self.n_iter):
            inverse = torch.istft(spec * angles, self.n_fft, self.hop_length,
                                  self.win_length, window=win, length=self.length)
            rebuilt = torch.stft(inverse, self.n_fft, self.hop_length, self.win_length,
                                 window=win, center=True, pad_mode="reflect",
                                 normalized=False, onesided=True, return_complex=True)
            angles = rebuilt - tprev * self.momentum
            tprev = rebuilt
            angles = angles / (angles.abs() + 1e-16)
        waveform = torch.istft(spec * angles, self.n_fft, self.hop_length,
                               self.win_length, window=win, length=self.length)
        return waveform.reshape(shape[:-2] + waveform.shape[-1:])


# ---------------------------------------------------------------------------
def _selftest():
    """Check the filterbank against librosa, which is an independent
    implementation already installed here (Resemblyzer depends on it)."""
    import numpy as np
    import librosa
    ok = True
    for (mel_scale, norm, htk, lnorm) in [("htk", None, True, None),
                                          ("slaney", "slaney", False, "slaney")]:
        ours = melscale_fbanks(513, 0.0, 8000.0, 80, 16000, norm=norm,
                               mel_scale=mel_scale).numpy().T
        theirs = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=80, fmin=0.0,
                                     fmax=8000.0, htk=htk, norm=lnorm)
        err = np.abs(ours - theirs).max()
        print(f"  mel_scale={mel_scale:7s} norm={str(norm):7s} max|ours-librosa| = {err:.3e}")
        ok &= err < 1e-5
    # round trip: mel -> lstsq -> mel should be close
    torch.manual_seed(0)
    wav = torch.randn(1, 16000) * 0.1
    ms = MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=80)
    inv = InverseMelScale(n_stft=513, n_mels=80, sample_rate=16000)
    mel = ms(wav)
    rec = inv(mel)
    remel = MelScale(n_mels=80, sample_rate=16000, n_stft=513)(rec)
    rel = ((remel - mel).norm() / mel.norm()).item()
    print(f"  mel -> InverseMelScale -> mel relative error = {rel:.3e}")
    gl = GriffinLim(n_fft=1024, n_iter=8, hop_length=256)
    print(f"  GriffinLim output shape = {tuple(gl(rec).shape)}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
