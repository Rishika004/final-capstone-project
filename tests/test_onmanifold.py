"""tests/test_onmanifold.py -- the shared on-manifold transforms (PLAN_V4).

Pins the three properties Step 0 and Step 1 depend on:

  * alpha = 1.0 is the analysis-resynthesis CONTROL: it must leave a voiced
    signal close to itself (it is the vocoder round trip, not a warp), while
    alpha = 0.8 must move it. If 1.0 and 0.8 were equally far from the input,
    the "control" would not be one.
  * Digital silence is handled explicitly and COUNTED (PREREG_step1 §2.5):
    a frame of exact zeros stays exactly zero and is reported in `stats`.
  * The transform is deterministic (the Step 1 gate's repeat check).

Run:  python -m pytest tests/test_onmanifold.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from utils.onmanifold import ARMS, apply_arm, arm_name, t_mcadams, t_tilt  # noqa: E402

SR = 16000


def _voiced(sec=1.0, f0=120.0, seed=0):
    """A crude vowel: pulse train through a fixed 3-formant resonator."""
    import scipy.signal
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    src = np.zeros(n)
    src[:: int(SR / f0)] = 1.0
    src += 0.01 * rng.standard_normal(n)
    y = src
    for fc, bw in ((700, 80), (1200, 90), (2500, 120)):
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * fc / SR
        y = scipy.signal.lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], y)
    y = y / np.abs(y).max() * 0.9
    return torch.from_numpy(y.astype(np.float32)).unsqueeze(0)


def _rel_change(a, b):
    a, b = a.squeeze().double(), b.squeeze().double()
    n = min(a.numel(), b.numel())
    return float((a[:n] - b[:n]).norm() / a[:n].norm())


def test_alpha_one_is_a_control_and_alpha_080_is_a_warp():
    x = _voiced()
    y1 = t_mcadams(x, 1.0)
    y8 = t_mcadams(x, 0.8)
    assert y1.shape == x.shape and y8.shape == x.shape
    assert _rel_change(x, y1) < 0.5 * _rel_change(x, y8), \
        "alpha=1.0 must be materially closer to the input than alpha=0.8"
    assert _rel_change(x, y8) > 0.2, "alpha=0.8 must actually move the signal"


def test_digital_silence_is_passed_through_and_counted():
    x = _voiced()
    x[:, 4000:12000] = 0.0            # 8000 exact zeros, the 121/260 hazard
    stats = {}
    y = t_mcadams(x, 0.8, stats=stats)
    assert stats["n_frames"] > 0
    assert stats["n_silent"] > 0, "silent frames must be counted, not hidden"
    # Frames entirely inside the zero run must come out exactly zero.
    assert torch.all(y[:, 5000:11000] == 0.0)
    assert torch.isfinite(y).all()


def test_mcadams_is_deterministic():
    x = _voiced()
    a = t_mcadams(x, 0.85)
    b = t_mcadams(x, 0.85)
    assert torch.equal(a, b)


def test_tilt_reports_stats_and_keeps_length():
    x = _voiced()
    win = torch.hann_window(1024)
    stats = {}
    y = t_tilt(x, -6.0, win, stats=stats)
    assert y.shape == x.shape
    assert stats["n_silent"] == 0 and stats["n_frames"] > 0


def test_arm_table_is_the_prereg_grid():
    assert set(ARMS) == {"mcadams_100", "mcadams_095", "mcadams_090",
                         "mcadams_085", "mcadams_080", "tilt_6db"}
    assert arm_name("mcadams", 0.8) == "mcadams_080"
    x = _voiced(0.5)
    for name in ARMS:
        st = {}
        y = apply_arm(name, x, stats=st)
        assert y.shape == x.shape and "n_frames" in st


def test_probe_wrapper_delegates_here():
    """exp_parametric_probe.t_mcadams must be this implementation."""
    pytest.importorskip("transformers")
    from experiments import exp_parametric_probe as probe
    x = _voiced(0.5)
    win = torch.hann_window(1024)
    assert torch.equal(probe.t_mcadams(x, 0.9, win), t_mcadams(x, 0.9, win))
