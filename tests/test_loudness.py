"""tests/test_loudness.py -- the loudness meter must actually be BS.1770.

A listening test normalised by a wrong meter is worse than one not normalised at
all: it looks controlled and is not. The standard's own reference case pins it.

Run:  python -m pytest tests/test_loudness.py -q
"""
import os
import sys

import numpy as np
import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from utils.loudness import integrated_lufs, normalise, k_weight

FS = 16000


def sine(db_fs, f=997.0, sec=5.0, fs=FS):
    amp = 10 ** (db_fs / 20.0)
    t = np.arange(int(sec * fs)) / fs
    return amp * np.sin(2 * np.pi * f * t)


# ------------------------------------------------------- the reference case

@pytest.mark.parametrize("db", [-20.0, -26.0, -13.0])
@pytest.mark.parametrize("fs", [48000, FS])
def test_mono_sine_reads_the_standard_value(db, fs):
    """BS.1770 reference: a MONO 997 Hz sine at X dBFS reads X - 3.00 LUFS.

    The -3 is real and not a bug: the standard's -0.691 offset is calibrated so
    that a 997 Hz tone in TWO channels reads its own level, and one channel
    carries half the summed power. A mono -20 dBFS tone is therefore -23.0 LUFS.

    Checked at 48 kHz, where the standard tabulates the filters, AND at our own
    16 kHz, which is the case that catches the filter-re-derivation error: the
    cookbook shelf gives +0.40 dB at 997 Hz against the standard's +0.691, so a
    native-rate 16 kHz meter reads -23.30 here. utils/loudness.py avoids that by
    resampling to 48 kHz and keeping the tabulated coefficients.
    """
    got = integrated_lufs(sine(db, fs=fs), fs)
    assert abs(got - (db - 3.0)) < 0.1, (
        f"997 Hz at {db} dBFS, fs={fs}, read {got:.3f} LUFS, "
        f"expected {db - 3.0:.2f}")


def test_the_meter_agrees_across_sample_rates():
    """The same tone must measure the same at 16 kHz and 48 kHz, or the
    listening test's normalisation depends on the file's rate."""
    a = integrated_lufs(sine(-20.0, fs=FS), FS)
    b = integrated_lufs(sine(-20.0, fs=48000), 48000)
    assert abs(a - b) < 0.05, f"16 kHz {a:.3f} vs 48 kHz {b:.3f}"


# The K-weighting response at the rate the standard tabulates it for, computed
# from the published coefficients. Pinned rather than bounded loosely: these are
# the numbers that make the meter the standard's meter and not merely a
# plausible high-shelf. +0.691 dB at 997 Hz is exactly the figure that produces
# the mono -3.00 LUFS identity above.
K_RESPONSE_48K = {20.0: 0.2169, 100.0: 0.8777, 997.0: 1.0828,
                  1681.97: 1.3309, 4000.0: 1.5791, 10000.0: 1.5926}


@pytest.mark.parametrize("f,expected", sorted(K_RESPONSE_48K.items()))
def test_k_weighting_matches_the_published_response(f, expected):
    """Magnitude response at `f`, measured as the RMS ratio of a pure tone."""
    def rms(x):
        return float(np.sqrt(np.mean(x ** 2)))
    x = sine(-20.0, f=f, sec=2.0, fs=48000)
    got = rms(k_weight(x, 48000)) / rms(x)
    assert abs(got - expected) < 0.01, (
        f"{f} Hz: K-weighting gain {got:.4f}, expected {expected:.4f}")


def test_louder_input_reads_louder():
    assert integrated_lufs(sine(-14.0), FS) > integrated_lufs(sine(-24.0), FS)


def test_doubling_amplitude_adds_six_db():
    x = sine(-26.0)
    assert abs((integrated_lufs(2 * x, FS) - integrated_lufs(x, FS)) - 6.02) < 0.05


# -------------------------------------------------------------- the gating

def test_silence_is_gated_out_not_averaged_in():
    """Half speech, half silence must read close to the speech alone.

    Without the absolute gate the silent half would halve the mean power and
    pull the reading down ~3 dB, which is exactly the error that makes sparse
    speech normalise too loud.
    """
    rng = np.random.default_rng(0)
    speech = 0.05 * rng.standard_normal(5 * FS)
    padded = np.concatenate([speech, np.zeros(5 * FS)])
    assert abs(integrated_lufs(padded, FS) - integrated_lufs(speech, FS)) < 0.6


def test_too_short_returns_minus_inf():
    assert integrated_lufs(np.zeros(100), FS) == float("-inf")


# --------------------------------------------------------------- normalise

def test_normalise_hits_the_target():
    for db in (-35.0, -20.0, -10.0):
        y, info = normalise(sine(db), FS, target_lufs=-23.0)
        assert abs(info["lufs_after"] - (-23.0)) < 0.1, info
        assert not info["clamped"], info


def test_normalise_reports_a_clamp_rather_than_silently_missing_target():
    """A quiet file that would need to exceed full scale must SAY so."""
    y, info = normalise(sine(-60.0), FS, target_lufs=0.0, peak_ceiling=0.5)
    assert info["clamped"] is True
    assert float(np.abs(y).max()) <= 0.5 + 1e-6


def test_normalise_preserves_waveform_shape():
    """Pure gain, no filtering: y/x must be one constant. (info["gain_db"] is
    rounded for the log, so test proportionality rather than that value.)"""
    x = sine(-30.0)
    y, _ = normalise(x, FS, target_lufs=-23.0)
    nz = np.abs(x) > 1e-6
    ratio = y[nz] / x[nz]
    assert np.allclose(ratio, ratio[0], rtol=1e-4), "normalise is not a pure gain"


def test_normalise_equalises_two_different_levels():
    """The point of the exercise: two conditions at different levels must end
    at the same loudness, so a listener cannot rate level instead of quality."""
    a, _ = normalise(sine(-30.0), FS, target_lufs=-23.0)
    b, _ = normalise(sine(-12.0), FS, target_lufs=-23.0)
    assert abs(integrated_lufs(a, FS) - integrated_lufs(b, FS)) < 0.1


def test_handles_stereo_shape():
    x = np.stack([sine(-20.0), sine(-20.0)])
    assert np.isfinite(integrated_lufs(x, FS))
