"""tests/test_matched_random.py -- the controls must be matched AND random.

Committed BEFORE the first #3 measurement (PLAN_V3 W1).  A control that quietly
fails to match its target on norm or band power would make alpha incomparable
and the gate meaningless; a control that accidentally correlates with its target
would hide a real effect.  Both failure modes are silent in the CSV, so they are
pinned here.

Run:  python -m pytest tests/test_matched_random.py -q
"""
import os
import sys

import pytest
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from core.matched_random import (band_matched, band_matched_wav, pca_span,
                                 controls_for, _per_band_rms, DEFAULT_K)

N_MELS, T = 80, 251          # the project's mel geometry (evaluation/evaluate.py)
RMS = 0.2                    # the protocol's per-element RMS


def _target(seed=0, bandlimit=None):
    """A stand-in encoder direction: structured, not white, optionally bandlimited."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn((1, N_MELS, T), generator=g)
    # Give it a spectral tilt so "matched on band power" is a real constraint.
    tilt = torch.linspace(3.0, 0.2, N_MELS).reshape(1, N_MELS, 1)
    d = d * tilt
    if bandlimit is not None:
        d[:, bandlimit:, :] = 0.0
    return d / d.pow(2).mean().sqrt() * RMS


def _basis(k=DEFAULT_K, seed=7):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn((N_MELS, k), generator=g))
    return q


# --------------------------------------------------------------- band_matched

def test_band_matched_preserves_norm_exactly():
    d = _target()
    c = band_matched(d, torch.Generator().manual_seed(1))
    assert torch.allclose(c.norm(), d.norm(), rtol=1e-5)


def test_band_matched_preserves_per_band_power_profile():
    d = _target()
    c = band_matched(d, torch.Generator().manual_seed(1))
    # The whole point of the control: same power in every mel band.
    assert torch.allclose(_per_band_rms(c), _per_band_rms(d), rtol=2e-3, atol=1e-6)


def test_band_matched_keeps_zeroed_bands_zero():
    """A bandlimited target must get a bandlimited control, or the control is
    strictly easier for a denoiser that only has to pass low frequencies."""
    d = _target(bandlimit=60)
    c = band_matched(d, torch.Generator().manual_seed(2))
    assert c[:, 60:, :].abs().max() == 0.0
    assert c[:, :60, :].abs().max() > 0.0


def test_band_matched_is_not_the_target_direction():
    """Matched on envelope, random in direction: cosine must be ~0, not ~1."""
    d = _target()
    c = band_matched(d, torch.Generator().manual_seed(3))
    cos = torch.nn.functional.cosine_similarity(
        c.reshape(1, -1), d.reshape(1, -1)).item()
    assert abs(cos) < 0.05, f"control correlates with its target (cos={cos:.3f})"


def test_band_matched_is_deterministic_given_a_seed():
    d = _target()
    a = band_matched(d, torch.Generator().manual_seed(4))
    b = band_matched(d, torch.Generator().manual_seed(4))
    assert torch.equal(a, b)
    c = band_matched(d, torch.Generator().manual_seed(5))
    assert not torch.equal(a, c)


# ------------------------------------------------------------------ pca_span

def test_pca_span_lies_in_the_basis_span():
    d, B = _target(), _basis()
    c = pca_span(B, d, torch.Generator().manual_seed(6))
    flat = c.reshape(N_MELS, T)
    resid = flat - B @ (B.T @ flat)          # component outside span(B)
    assert (resid.norm() / flat.norm()).item() < 1e-5


def test_pca_span_preserves_norm_exactly():
    d, B = _target(), _basis()
    c = pca_span(B, d, torch.Generator().manual_seed(7))
    assert torch.allclose(c.norm(), d.norm(), rtol=1e-5)


def test_pca_span_shape_matches_target():
    d, B = _target(), _basis()
    assert pca_span(B, d, torch.Generator().manual_seed(8)).shape == d.shape


def test_pca_span_rank_is_the_basis_rank():
    """k components in, rank k out -- a control that silently spanned all 80
    mels would not be the low-dimensional speech subspace we claim to test."""
    d, B = _target(), _basis(k=12)
    c = pca_span(B, d, torch.Generator().manual_seed(9))
    sv = torch.linalg.svdvals(c.reshape(N_MELS, T))
    assert int((sv > sv[0] * 1e-6).sum()) == 12


# ----------------------------------------------------------- controls_for

def test_controls_for_returns_both_controls_norm_matched():
    d, B = _target(), _basis()
    out = controls_for(d, B, seed=61)
    assert set(out) == {"band_matched", "pca_span"}
    for name, c in out.items():
        assert torch.allclose(c.norm(), d.norm(), rtol=1e-5), name
        assert not c.requires_grad, name


def test_controls_for_without_a_basis_still_gives_band_matched():
    d = _target()
    assert set(controls_for(d, None, seed=61)) == {"band_matched"}


def test_controls_for_is_reproducible_from_the_speaker_id():
    d, B = _target(), _basis()
    a = controls_for(d, B, seed=1272)
    b = controls_for(d, B, seed=1272)
    c = controls_for(d, B, seed=1273)
    for kind in a:
        assert torch.equal(a[kind], b[kind]), kind
        assert not torch.equal(a[kind], c[kind]), kind


def test_the_two_controls_are_independent_draws():
    d, B = _target(), _basis()
    out = controls_for(d, B, seed=61)
    cos = torch.nn.functional.cosine_similarity(
        out["band_matched"].reshape(1, -1), out["pca_span"].reshape(1, -1)).item()
    assert abs(cos) < 0.2, f"the two controls coincide (cos={cos:.3f})"


# -------------------------------------------------------- basis construction

@pytest.mark.skipif(
    not os.path.isdir(os.path.join(PROJ, "data", "librispeech", "LibriSpeech", "dev-clean")),
    reason="LibriSpeech dev-clean not on disk")
def test_clean_mel_basis_is_orthonormal_and_caches():
    from core.matched_random import clean_mel_basis
    B, info = clean_mel_basis(k=8, n_files=3, cache=None, verbose=False)
    assert B.shape == (N_MELS, 8)
    assert torch.allclose(B.T @ B, torch.eye(8), atol=1e-4)
    # Top components of clean log-mel must explain a large share of variance;
    # if they did not, "the clean-speech principal subspace" would be a misnomer.
    assert 0.5 < info["var_explained"] <= 1.0


# ----------------------------------------------- what the controls do NOT match
# RESULTS 58, risk 1. Both controls match a time-AVERAGED spectral profile and
# randomise everything else, so in time they are as flat as white noise
# whatever the target looks like. band_matched ~ random in alpha therefore means
# "matching the spectrum did not help the control", and NOT "the control is as
# structured as the target". The two tests below pin that property so nobody
# reads the control as more than it is, and so that a control that also
# matches temporal structure has to change them knowingly.

def _frame_cv(x):
    e = x.reshape(N_MELS, T).pow(2).sum(dim=0)
    return float(e.std() / e.mean())


def test_band_matched_is_temporally_white_whatever_the_target():
    d = _target()
    d[..., : T // 2] = 0.0                       # concentrated in time, like a gradient
    d = d / d.pow(2).mean().sqrt() * RMS
    c = band_matched(d, torch.Generator().manual_seed(11))
    assert _frame_cv(d) > 0.9
    assert _frame_cv(c) < 0.35, _frame_cv(c)    # white noise sits near 0.16


def test_band_matched_wav_is_gaussian_in_time_whatever_the_target():
    g = torch.Generator().manual_seed(12)
    d = torch.randn(64000, generator=g)
    d[torch.rand(64000, generator=g) > 0.03] = 0.0      # a sparse, spiky target
    kurt = lambda x: float(((x - x.mean()) ** 4).mean() / x.var() ** 2)  # noqa: E731
    assert kurt(d) > 20
    c = band_matched_wav(d, generator=torch.Generator().manual_seed(13))
    assert 2.6 < kurt(c) < 3.4, kurt(c)                 # Gaussian: 3
    assert torch.allclose(c.norm(), d.norm(), rtol=1e-5)
