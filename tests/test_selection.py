"""tests/test_selection.py -- GOLDEN FILE for the v1 crafting objective.

PLAN_V3 W1 requires this test to be committed BEFORE the 2.1 / 2.2 / 2.3 / 2d /
2e code fixes, because those fixes touch the two expressions that decide what
the optimiser minimises and which iterate gets shipped. Every arm in the paper
was crafted by the legacy forms of those expressions, PLAN_V3's base rule is
"the base never moves", and `recipe-v1` has to stay reproducible from HEAD.

So this file pins the LEGACY behaviour numerically. The pinned expressions are
the pre-refactor closures of `craft()`:

    lse(c)   = logsumexp((c - thr_t) * w * tau_lse) / tau_lse
    score    = c.max().item()

evaluated with the shipping recipe's own constants: tau_lse 4.0, weights 1,1,1,
and the thresholds the m30_seg arms were actually crafted against --
`thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv`, i.e. 0.8740 / 0.3067 /
0.6819 (NOT the clean-channel triple 0.8926 / 0.3358 / 0.7029, which is what
the cloning script scores with; mixing the two is the §23c/§31 confusion).

If a later edit changes what those produce, this test fails and the edit has
silently re-defined the crafting objective -- the RESULTS §25 failure mode, one
level up.

The second half asserts the FIXES are off by default, and that when switched on
they change the decision in the documented direction: raw-cosine selection can
prefer an iterate whose worst threshold margin is worse (2.1), and dividing by
each encoder's own score scale can move which encoder binds (2e, RESULTS 35).
Those are the premises of the fixes; if they never disagreed the fixes would be
pointless and PLAN_V3's §5 disclosure paragraph would be unnecessary.

Run:  python -m pytest tests/test_selection.py -q
"""
import os
import sys

import pytest
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# ---------------------------------------------------------------- constants
TAU_LSE = 4.0
WEIGHTS = (1.0, 1.0, 1.0)
# The crafting calibration, in build_encoders() order:
#   WavLM-SV, ECAPA-TDNN, Resemblyzer d-vector
THR = (0.8740, 0.3067, 0.6819)
# Per-encoder margin scales for 2e, as utils.thresholds.encoder_scales returns
# them for this channel (inferred from the EER: this calibration predates the
# target_sd / nontarget_sd columns that PLAN_V3 2f added).
SCALES = (0.08960, 0.09955, 0.06524)


def _import():
    """exp_purifier_aware imports torch._dynamo / transformers / speechbrain at
    module level; skip rather than fail if that environment is unavailable."""
    try:
        from experiments.exp_purifier_aware import agg_lse, ship_score
    except Exception as exc:                        # pragma: no cover
        pytest.skip(f"exp_purifier_aware not importable here: {exc}")
    return agg_lse, ship_score


# Three fixed cosine vectors spanning the regimes the optimiser passes through:
# well above threshold, straddling it, and below it.
CASES = {
    "above":    (0.95, 0.60, 0.80),
    "straddle": (0.88, 0.34, 0.72),
    "below":    (0.70, 0.10, 0.55),
}

# Golden values of the LEGACY expressions on those inputs.
LEGACY_LSE = {
    "above":    0.4557914,
    "straddle": 0.3008478,
    "below":    0.1056693,
}
LEGACY_SCORE = {"above": 0.95, "straddle": 0.88, "below": 0.70}


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


# ============================================================ the golden file
@pytest.mark.parametrize("key", sorted(CASES))
def test_legacy_lse_is_unchanged(key):
    """The ensemble objective must still compute what crafted recipe-v1."""
    agg_lse, _ = _import()
    got = float(agg_lse(_t(CASES[key]), _t(THR), _t(WEIGHTS), TAU_LSE,
                        scales=None))
    assert got == pytest.approx(LEGACY_LSE[key], abs=1e-6), (
        f"the v1 ensemble objective moved on case {key!r}: "
        f"{got:.7f} != {LEGACY_LSE[key]:.7f}")


@pytest.mark.parametrize("key", sorted(CASES))
def test_legacy_lse_matches_the_original_expression(key):
    """Belt and braces: compare against the literal pre-refactor expression, so
    the golden constants cannot drift together with the code."""
    agg_lse, _ = _import()
    c, thr, w = _t(CASES[key]), _t(THR), _t(WEIGHTS)
    original = torch.logsumexp((c - thr) * w * TAU_LSE, dim=0) / TAU_LSE
    assert float(agg_lse(c, thr, w, TAU_LSE, scales=None)) == pytest.approx(
        float(original), abs=1e-12)


@pytest.mark.parametrize("key", sorted(CASES))
def test_legacy_ship_score_is_the_max_raw_cosine(key):
    """v1 selected on the MAX RAW COSINE, blind to the per-encoder threshold."""
    _, ship_score = _import()
    got = ship_score(_t(CASES[key]), _t(THR), scales=None, legacy=True)
    assert got == pytest.approx(LEGACY_SCORE[key], abs=1e-9)


def test_weights_still_enter_the_objective():
    """The 1,1,1.5 default weighting is part of the objective; pin that it is
    applied inside the logsumexp rather than outside."""
    agg_lse, _ = _import()
    c, thr = _t(CASES["straddle"]), _t(THR)
    w = _t((1.0, 1.0, 1.5))
    expect = torch.logsumexp((c - thr) * w * TAU_LSE, dim=0) / TAU_LSE
    assert float(agg_lse(c, thr, w, TAU_LSE)) == pytest.approx(float(expect),
                                                               abs=1e-12)


def test_fixes_are_off_by_default():
    """A bare re-run from HEAD must reproduce v1: legacy is the default both on
    the helper and on craft(), and 2.3's knobs keep their crafted values."""
    import inspect
    _, ship_score = _import()
    from experiments.exp_purifier_aware import craft

    assert ship_score(_t(CASES["straddle"]), _t(THR)) == pytest.approx(
        LEGACY_SCORE["straddle"])

    sig = inspect.signature(craft)
    assert sig.parameters["legacy_selection"].default is True
    assert sig.parameters["enc_scales"].default is None
    assert sig.parameters["tau_lse"].default == 4.0
    assert sig.parameters["seg_tau"].default == 8.0


def test_scales_are_ignored_on_the_legacy_path():
    """Passing scales must not change v1 behaviour unless the fix is enabled."""
    _, ship_score = _import()
    assert ship_score(_t(CASES["straddle"]), _t(THR), scales=_t(SCALES),
                      legacy=True) == pytest.approx(LEGACY_SCORE["straddle"])


# ====================================================== the fixes' premises
def test_margin_selection_can_disagree_with_raw_cosine():
    """2.1's premise. The crafting thresholds span 0.567 (0.3067 to 0.8740), so
    the encoder with the largest raw cosine need not be the one closest to
    accepting, and the RANKING of two candidate iterates can invert."""
    _, ship_score = _import()
    thr = _t(THR)

    c = _t((0.85, 0.50, 0.60))
    assert ship_score(c, thr, legacy=True) == pytest.approx(0.85)    # WavLM-SV
    assert ship_score(c, thr, legacy=False) == pytest.approx(0.1933)  # ECAPA

    # Two iterates: raw cosine prefers `a`, the margin prefers `b`.
    a, b = _t((0.85, 0.50, 0.60)), _t((0.88, 0.33, 0.60))
    assert ship_score(a, thr, legacy=True) < ship_score(b, thr, legacy=True)
    assert ship_score(a, thr, legacy=False) > ship_score(b, thr, legacy=False)


def test_per_encoder_scale_can_move_the_binding_encoder():
    """2e / RESULTS 35: raw margins are not commensurable across encoders.

    With this channel's own measured scales, a case where ECAPA binds on raw
    margins has the d-vector binding once each margin is in units of that
    encoder's own score spread. RESULTS 35a measured this flip for 4 of 29
    speakers on real crafted files.
    """
    _, ship_score = _import()
    thr, scales = _t(THR), _t(SCALES)
    c = _t((0.8790, 0.3467, 0.7119))        # margins +0.005, +0.040, +0.030

    raw = c - thr
    assert int(raw.argmax()) == 1                      # ECAPA binds
    assert ship_score(c, thr, legacy=False) == pytest.approx(0.040, abs=1e-6)

    scaled = raw / scales
    assert int(scaled.argmax()) == 2                   # the d-vector binds
    assert ship_score(c, thr, scales=scales, legacy=False) == pytest.approx(
        0.459841, abs=1e-5)


def test_trailing_mean_selection_rejects_a_lucky_single_draw():
    """2d, as a property of the selection rule rather than of craft()'s loop.

    At n_eot=1 each step's score is ONE purification draw, so argmin over 200
    steps ships the luckiest draw. A trajectory that is uniformly mediocre but
    contains one fortunate draw beats a uniformly better trajectory under
    argmin, and loses under a trailing mean. That inversion is the fix.
    """
    window = 5

    def legacy_pick(scores):
        best = float("inf")
        for s in scores:
            best = min(best, s)
        return best

    def sma_pick(scores, w=window):
        best, hist = float("inf"), []
        for s in scores:
            hist.append(s)
            if len(hist) > w:
                hist.pop(0)
            if len(hist) == w:
                best = min(best, sum(hist) / w)
        return best

    lucky = [0.50] * 4 + [0.10] + [0.50] * 5        # one fortunate draw
    steady = [0.30] * 10                            # genuinely better
    assert legacy_pick(lucky) < legacy_pick(steady)
    assert sma_pick(lucky) > sma_pick(steady)
