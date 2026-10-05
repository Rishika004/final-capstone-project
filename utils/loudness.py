"""utils/loudness.py -- ITU-R BS.1770-4 integrated loudness and normalisation.

Needed for the listening test (council #6, PLAN_V3 W1): the existing stimuli are
not loudness-normalised, so a listener rating "naturalness" is partly rating
level. Protected files are quieter than originals by construction -- the method
is an SNR-constrained perturbation -- so level is CONFOUNDED WITH CONDITION,
which is the one confound a MOS test must not have.

Implemented here rather than taking a dependency (pyloudnorm is not installed).

HOW IT STAYS COMPLIANT
----------------------
BS.1770 tabulates its two K-weighting biquads at 48 kHz only. Re-deriving them
at another rate from the cookbook high-shelf formulas does NOT reproduce the
standard's stage-1 response: at 16 kHz that route gives +0.40 dB at 997 Hz where
the standard gives +0.691 dB, so a mono 997 Hz tone at -20 dBFS reads -23.30
LUFS instead of the standard's -23.00. Measured, not assumed -- see the git
history of this file.

So this module does the opposite: it keeps the standard's exact 48 kHz
coefficients and resamples the signal to 48 kHz to measure it. Resampling
preserves loudness, 16 kHz to 48 kHz is an exact x3, and the meter is then
compliant at every input rate rather than only at one.

Algorithm (BS.1770-4):
  1. K-weighting: high-shelf (+4 dB, 1681.97 Hz) then RLB high-pass (38.13 Hz).
  2. Mean square per 400 ms block, 75 % overlap.
  3. Two-stage gating: absolute at -70 LUFS, then relative at -10 LU below the
     mean of the blocks surviving the absolute gate.
  4. Loudness = -0.691 + 10 log10(mean surviving block power).
"""
import numpy as np

# BS.1770-4 tabulated coefficients, 48 kHz. Do not "simplify" these.
FS_REF = 48000
_SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_HPF_B = np.array([1.0, -2.0, 1.0])
_HPF_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

BLOCK_S = 0.400
OVERLAP = 0.75
ABS_GATE_LUFS = -70.0
REL_GATE_LU = -10.0
OFFSET_DB = -0.691


def _lfilter(b, a, x):
    try:
        from scipy.signal import lfilter
        return lfilter(b, a, x)
    except ImportError:                      # direct form I fallback
        y = np.zeros_like(x, dtype=np.float64)
        x1 = x2 = y1 = y2 = 0.0
        for i, xn in enumerate(x):
            yn = b[0] * xn + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            x2, x1 = x1, xn
            y2, y1 = y1, yn
            y[i] = yn
        return y


def _to_48k(x, fs):
    """Resample to 48 kHz for measurement. Loudness is preserved."""
    if fs == FS_REF:
        return x, FS_REF
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(FS_REF), int(fs))
        return resample_poly(x, FS_REF // g, fs // g, axis=-1), FS_REF
    except ImportError:
        # No scipy: measure at the native rate with re-derived filters. The
        # absolute reading may differ from the standard by a few tenths of a dB;
        # relative normalisation between files is unaffected.
        return x, fs


def _filters(fs):
    if fs == FS_REF:
        return (_SHELF_B, _SHELF_A), (_HPF_B, _HPF_A)
    # Cookbook re-derivation, only reached without scipy. See the module note.
    f0, G_db, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    A = 10 ** (G_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    al, sqA = sw / (2 * Q), np.sqrt(A)
    sb = np.array([A * ((A + 1) + (A - 1) * cw + 2 * sqA * al),
                   -2 * A * ((A - 1) + (A + 1) * cw),
                   A * ((A + 1) + (A - 1) * cw - 2 * sqA * al)])
    sa = np.array([(A + 1) - (A - 1) * cw + 2 * sqA * al,
                   2 * ((A - 1) - (A + 1) * cw),
                   (A + 1) - (A - 1) * cw - 2 * sqA * al])
    f0, Q = 38.13547087602444, 0.5003270373238773
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    al = sw / (2 * Q)
    hb = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    ha = np.array([1 + al, -2 * cw, 1 - al])
    return (sb / sa[0], sa / sa[0]), (hb / ha[0], ha / ha[0])


def k_weight(x, fs):
    """Apply the two BS.1770 K-weighting stages at the signal's own rate."""
    (sb, sa), (hb, ha) = _filters(fs)
    return _lfilter(hb, ha, _lfilter(sb, sa, np.asarray(x, dtype=np.float64)))


def integrated_lufs(x, fs):
    """Integrated (gated) loudness in LUFS. Mono 1-D, or [channels, n]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    elif x.shape[0] > x.shape[1]:
        x = x.T

    x, fs = _to_48k(x, fs)
    n = x.shape[-1]
    block = int(round(BLOCK_S * fs))
    if n < block:
        return float("-inf")
    step = max(int(round(block * (1 - OVERLAP))), 1)

    weighted = [k_weight(ch, fs) for ch in x]      # channel weights are 1.0 for L/R/C
    starts = range(0, n - block + 1, step)
    powers = np.array([sum(float(np.mean(w[s:s + block] ** 2)) for w in weighted)
                       for s in starts])

    with np.errstate(divide="ignore"):
        lj = OFFSET_DB + 10 * np.log10(np.maximum(powers, 1e-300))
    keep = lj > ABS_GATE_LUFS
    if not keep.any():
        return float("-inf")
    rel = OFFSET_DB + 10 * np.log10(powers[keep].mean()) + REL_GATE_LU
    keep = keep & (lj > rel)
    if not keep.any():
        return float("-inf")
    return float(OFFSET_DB + 10 * np.log10(powers[keep].mean()))


def normalise(x, fs, target_lufs=-23.0, peak_ceiling=0.99):
    """Scale `x` to `target_lufs`.

    Returns (scaled, info). `info` carries the measured input loudness, the gain
    applied, the achieved loudness, and whether the peak ceiling clamped it. A
    clamp means the file did NOT reach the target, and it is reported rather
    than shipped silently -- otherwise the normalisation the listening test
    depends on is simply absent for that file.
    """
    x = np.asarray(x, dtype=np.float64)
    before = integrated_lufs(x, fs)
    if not np.isfinite(before):
        return x.astype(np.float32), dict(
            lufs_before=before, gain_db=0.0, lufs_after=before,
            clamped=False, peak=float(np.abs(x).max()) if x.size else 0.0)
    gain_db = float(target_lufs - before)
    y = x * 10 ** (gain_db / 20.0)
    peak = float(np.abs(y).max())
    clamped = peak > peak_ceiling
    if clamped:
        y = y * (peak_ceiling / peak)
    return y.astype(np.float32), dict(
        lufs_before=round(before, 2), gain_db=round(gain_db, 2),
        lufs_after=round(integrated_lufs(y, fs), 2), clamped=clamped,
        peak=round(float(np.abs(y).max()), 4))
