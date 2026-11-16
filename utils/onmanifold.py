"""utils/onmanifold.py -- the on-manifold transforms, in ONE place.

PLAN_V4 Step 0 (listening stimuli) and Step 1 (the warp re-measurement) both
apply McAdams formant shifting. Until 2026-10-10 the only implementation lived
inside `experiments/exp_parametric_probe.py`, next to the September probe it
was written for. Two consumers importing from a probe script is how two code
paths drift apart by accident (RESULTS 25, 37e), so the transform now lives
here and the probe imports it.

`t_mcadams` is the September implementation, recovered byte-for-byte in
behaviour from `experiments/__pycache__/exp_parametric_probe.cpython-312.pyc`
(the source had already been replaced by an import of this file, which did not
yet exist). Nothing in the arithmetic changed. What was added:

  * **Explicit digital-silence handling, counted and reported.** The old code
    already passed a frame through unchanged when its peak was below 1e-7 --
    LPC on exact zeros has a singular autocorrelation matrix -- but it did so
    silently. PREREG_step1 §2.5 names speakers 121 and 260 (8637 and 8794
    consecutive zero samples) as the cohort hazard and requires the handling to
    be *reported, not patched silently*. `stats=` now returns the counts of
    silent, failed (LPC raised) and unstable (modified filter had a pole on or
    outside the unit circle, fell back to the unmodified filter) frames.
  * A `stats` keyword on every transform so the Step 1 harness can record them
    on each row next to `gate_passed`.

The McAdams transform (Patino et al., arXiv 2011.01130; VoicePrivacy B2):
per frame, LPC of order 16; each complex pole's phase phi -> sign(phi)|phi|^alpha;
the LPC residual is filtered through the modified all-pole filter; frames are
overlap-added with a Hann window. alpha = 1.0 is the analysis-resynthesis
control (no formant shift, the vocoder round trip alone).
"""
import numpy as np
import torch

SILENT_PEAK = 1e-7          # a frame below this is digital silence: passed through


def _stft(wav, win, n_fft, hop):
    return torch.stft(wav.squeeze(0), n_fft, hop, window=win, return_complex=True)


def _istft(spec, win, n_fft, hop, length):
    return torch.istft(spec, n_fft, hop, window=win, length=length).unsqueeze(0)


def t_tilt(wav, db, win, n_fft=1024, hop=256, stats=None, **kw):
    """Broadband spectral tilt: `db` dB difference between DC and Nyquist,
    applied to the STFT magnitude with the original phase. The §8.2 existence
    proof: free, purification-invariant, 0 % protective."""
    F_ = n_fft // 2 + 1
    f = torch.linspace(0, 1, F_, device=wav.device)
    gain = 10 ** (db * f / 20.0)
    S = _stft(wav, win, n_fft, hop)
    y = _istft(S * gain.unsqueeze(1), win, n_fft, hop, wav.shape[-1])
    if stats is not None:
        stats.update(n_frames=int(S.shape[-1]), n_silent=0, n_failed=0, n_unstable=0)
    return y


def t_mcadams(wav, alpha, win=None, frame_len=320, hop=160, order=16,
              stats=None, **kw):
    """McAdams-coefficient formant shifting. See the module docstring.

    `wav` is [1, T] float at 16 kHz. `win` is accepted for signature parity with
    the STFT transforms and ignored. Returns [1, T] on `wav.device`, peak-matched
    to the input (the September behaviour).

    `stats`, if a dict, receives n_frames / n_silent / n_failed / n_unstable.
    """
    import scipy.signal
    import librosa

    x = wav.squeeze(0).detach().cpu().numpy().astype(np.float64)
    n = x.shape[0]
    w = np.hanning(frame_len)
    out = np.zeros(n + frame_len)
    nrm = np.zeros(n + frame_len) + 1e-12
    n_frames = n_silent = n_failed = n_unstable = 0
    for st in range(0, max(n - frame_len, 1), hop):
        n_frames += 1
        seg = x[st:st + frame_len] * w
        if seg.shape[0] < frame_len or np.abs(seg).max() < SILENT_PEAK:
            # Digital silence (or a ragged tail): LPC has nothing to fit and the
            # autocorrelation matrix is singular. Pass the frame through.
            n_silent += 1
            out[st:st + frame_len] += seg
            nrm[st:st + frame_len] += w ** 2
            continue
        try:
            a = librosa.lpc(seg, order=order)
        except Exception:
            n_failed += 1
            out[st:st + frame_len] += seg
            nrm[st:st + frame_len] += w ** 2
            continue
        res = scipy.signal.lfilter(a, [1.0], seg)
        poles = np.roots(a)
        newp = []
        for pz in poles:
            if abs(np.imag(pz)) > 1e-12:
                phi = np.angle(pz)
                newp.append(np.abs(pz) * np.exp(1j * np.sign(phi) * np.abs(phi) ** alpha))
            else:
                newp.append(pz)
        a_new = np.real(np.poly(np.array(newp)))
        if not np.all(np.abs(np.roots(a_new)) < 1.0):
            n_unstable += 1
            a_new = a
        seg_new = scipy.signal.lfilter([1.0], a_new, res)
        out[st:st + frame_len] += seg_new * w
        nrm[st:st + frame_len] += w ** 2
    y = (out / nrm)[:n]
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.abs(y).max()
    if peak > 1e-8:
        y = y / peak * float(wav.abs().max())
    if stats is not None:
        stats.update(n_frames=n_frames, n_silent=n_silent, n_failed=n_failed,
                     n_unstable=n_unstable)
    return torch.from_numpy(y).float().unsqueeze(0).to(wav.device)


# The Step 0 / Step 1 condition table: one spelling of each arm name, shared by
# the stimulus builder, the Step 1 harness and the analysis scripts.
#   name -> (transform, parameter)
MCADAMS_ALPHAS = (1.00, 0.95, 0.90, 0.85, 0.80)
TILT_DB = -6.0


def arm_name(kind, param):
    if kind == "mcadams":
        return f"mcadams_{int(round(param * 100)):03d}"
    if kind == "tilt":
        return f"tilt_{int(round(abs(param)))}db"
    raise ValueError(kind)


ARMS = {arm_name("mcadams", a): ("mcadams", a) for a in MCADAMS_ALPHAS}
ARMS[arm_name("tilt", TILT_DB)] = ("tilt", TILT_DB)


def apply_arm(name, wav, win=None, stats=None):
    kind, param = ARMS[name]
    if kind == "mcadams":
        return t_mcadams(wav, param, win, stats=stats)
    if kind == "tilt":
        if win is None:
            win = torch.hann_window(1024, device=wav.device)
        return t_tilt(wav, param, win, stats=stats)
    raise ValueError(kind)
