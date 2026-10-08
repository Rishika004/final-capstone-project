"""tests/test_direction_gate.py -- the instrument check must reject residue and
accept a gradient (RESULTS 57f's rule, re-audited in RESULTS 58).

Model-free: a toy L2-normalised encoder f(x) = normalize(tanh(A x) B) stands in
for every speaker encoder, because the property under test is a property of
L2 normalisation, not of any network:

    c(x) = <f(x), f(x0)>  is maximal at x0, so grad c(x0) = 0 analytically.

Evaluated AT x0 (what #13c did), the float32 gradient is rounding residue and
the gate must say so. Evaluated at a reconstruction point x_r != x0 against
f(x0) (what the mel path does), the gradient is genuine and the gate must pass
it. The float64 test pins the mathematics directly.

Run:  python -m pytest tests/test_direction_gate.py -q
"""
import os
import sys

import pytest
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.check_direction_sanity import check_direction, scale_to_rms  # noqa: E402

N_IN, N_HID, N_OUT = 1000, 64, 32


def _toy(dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn((N_HID, N_IN), generator=g) / N_IN ** 0.5
    B = torch.randn((N_HID, N_OUT), generator=g) / N_HID ** 0.5
    A, B = A.to(dtype), B.to(dtype)

    def f(x):
        return torch.nn.functional.normalize((torch.tanh(A @ x) @ B).reshape(1, -1), dim=-1)
    x0 = torch.randn(N_IN, generator=g).to(dtype)
    off = torch.randn(N_IN, generator=g).to(dtype)
    off = off / off.norm() * 0.5 * x0.norm()          # a fixed, sizeable offset
    return f, x0, off


def _noise(n, seed, dtype):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed)).to(dtype)


def _grad_at(f, x_eval, o):
    w = x_eval.clone().requires_grad_(True)
    c = (f(w) * o).sum()
    g, = torch.autograd.grad(c, w)
    return (-g).detach(), float(c), float(g.norm())


# --------------------------------------------------------------- mathematics

def test_gradient_of_cosine_to_self_vanishes_at_the_signal_in_float64():
    f, x0, off = _toy(torch.float64)
    o = f(x0).detach()
    _, c_at, g_at = _grad_at(f, x0, o)
    _, c_off, g_off = _grad_at(f, x0 + off, o)
    assert abs(c_at - 1.0) < 1e-12
    assert c_off < 0.999, "the offset point must sit below the maximum for this test to mean anything"
    # 10+ orders of magnitude between the analytic zero and a real gradient
    assert g_at < 1e-10 * g_off, (g_at, g_off)


def test_float32_gradient_at_the_signal_is_residue_not_a_direction():
    f32, x0, off = _toy(torch.float32)
    f64, x064, _ = _toy(torch.float64)
    d32, _, g32 = _grad_at(f32, x0, f32(x0).detach())
    d64, _, g64 = _grad_at(f64, x064, f64(x064).detach())
    # float32 leaves something non-zero; float64 does not; they are unrelated
    assert g32 > 0 and g64 < 1e-6 * g32
    if g64 > 0:
        cos = torch.nn.functional.cosine_similarity(d32.double().reshape(1, -1),
                                                    d64.reshape(1, -1)).item()
        assert abs(cos) < 0.5


# ---------------------------------------------------------------- the gate

def test_gate_rejects_the_direction_taken_at_the_maximum():
    """What exp_encoder_subspace_wav.py did (RESULTS 55 -> 57b)."""
    f, x0, _ = _toy(torch.float32)
    rms = x0.pow(2).mean().sqrt()

    def at_max(jitter):
        x_in = x0 + jitter * rms * _noise(N_IN, 1, torch.float32)
        return _grad_at(f, x_in, f(x_in).detach())       # o recomputed: still the maximum

    ok, report = check_direction(at_max, name="toy at x0")
    assert not ok
    assert "STATIONARY" in report


def test_gate_accepts_a_direction_taken_off_the_maximum_at_several_jitters():
    """What exp_encoder_subspace.py does: evaluate at a reconstruction point."""
    f, x0, off = _toy(torch.float32)
    rms = x0.pow(2).mean().sqrt()

    def off_max(jitter):
        x_in = x0 + jitter * rms * _noise(N_IN, 1, torch.float32)
        return _grad_at(f, x_in + off, f(x_in).detach())

    for j in (1e-5, 1e-4, 1e-3):
        ok, report = check_direction(off_max, name="toy at x_r", jitter=j)
        assert ok, report


def test_gate_names_a_sign_flip():
    f, x0, off = _toy(torch.float32)

    def flips(jitter):
        d, c, n = _grad_at(f, x0 + off, f(x0).detach())
        return (-d if jitter > 0 else d), c, n

    ok, report = check_direction(flips, name="toy flip")
    assert not ok
    assert "FLIPS SIGN" in report


# --------------------------------------------------------------- scale_to_rms

def test_scale_to_rms_is_exact_for_a_real_direction():
    v = torch.randn((1, 80, 251), generator=torch.Generator().manual_seed(3)) * 1e-4
    out = scale_to_rms(v, 0.2)
    assert abs(float(out.pow(2).mean().sqrt()) - 0.2) < 1e-6
    assert torch.allclose(out / out.norm(), v / v.norm())


def test_scale_to_rms_refuses_residue():
    """#13c's d-vector arm: RMS 4.6e-11 against an additive epsilon of 1e-12
    lost 1.2 % of its RMS silently. The scaler must refuse, not rescale."""
    v = torch.randn(64000, generator=torch.Generator().manual_seed(4)) * 4.6e-11
    with pytest.raises(ValueError):
        scale_to_rms(v, 0.05)
