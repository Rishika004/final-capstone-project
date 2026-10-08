"""
experiments/analyse_tost.py
===========================
RESULTS 30 as an EQUIVALENCE test rather than a failure to reject.

Council v4 E2. R3 calls this "the highest-value statistic" left in the project,
and the reason is a logical gap, not a taste in methods.

THE GAP
-------
RESULTS 30 reports "1 of 21 Holm cells significant" and concludes that BPDA and
exact gradients give the same protection. That inference is invalid in the
direction it is used: a non-rejection is not evidence of equivalence, it is an
absence of evidence of difference. And the conclusion is LOAD-BEARING --

  * `FRAME_v6` C3.2 states it as a contribution ("the gradient estimator is not
    the lever"),
  * the #5 dev-clean confirmatory run was crafted with `--grad-estimator bpda`
    ON THE STRENGTH OF IT (`run_devclean_replication.sh`), so if the estimators
    are NOT equivalent the headline replication is on a different recipe from
    the headline it replicates.

So the claim has to be tested as what it is: an equivalence claim.

THE TEST
--------
Two one-sided tests (Schuirmann). For each cell, with paired per-speaker
differences d = exact - BPDA and an equivalence margin D:

    H01: mu <= -D      H02: mu >= +D
    reject BOTH  ->  equivalence at margin D

Implemented as the standard TOST-CI identity: equivalence at level alpha holds
iff the (1 - 2*alpha) CI for mu lies entirely inside (-D, +D). We use the
percentile bootstrap on the paired differences for the interval, to match the
rest of the project, and report the exact-t TOST p-values beside it.

THE MARGIN, AND WHY IT IS 0.01
------------------------------
D = 0.01 was fixed in `PLAN_V3` W8 ("TOST |delta| < 0.01 for 30"), written
2026-10-05, BEFORE this analysis -- so it is pre-specified rather than tuned to
the data, which is the whole point of an equivalence margin.

It is also defensible on its own terms:
  * the smallest effect any claim in the paper rests on is 0.014 (RESULTS 49's
    d-vector t=3 cells), so 0.01 is below the paper's own resolution;
  * RESULTS 34's empirical floor for a crafting-loop edit is 0.0086-0.0557
    paired SD, and 0.01 sits at the bottom of that range;
  * RESULTS 50 measured a run-to-run term of up to 0.03 on one utterance with
    identical code, and BPDA-vs-exact is a comparison of two separate crafting
    runs, so part of any difference found here is rerun noise.

The last point cuts in the SAFE direction for an equivalence claim: extra
variance widens the CI and makes equivalence HARDER to establish. If equivalence
holds despite it, it holds.

The script also reports, per cell, the SMALLEST margin at which equivalence
would hold (the "achieved bound"), which is more informative than a verdict at
one margin and cannot be gamed after the fact because the pre-specified verdict
is printed first.

Run:
    python experiments/analyse_tost.py
    python experiments/analyse_tost.py --margin 0.02
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.analyse_matched_control import holm

R = os.path.join(PROJ, "experiments", "results")
ENC_ORDER = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]

# (label, exact file, bpda file) for every channel RESULTS 30 reports.
# The constrained arm is the one the confirmatory runs use.
CHANNELS = [
    ("DiffWave-LibriSpeech t=1", "protect_foreign_seg_libri_t1.csv",
     "protect_foreign_bpda_libri_t1.csv"),
    ("DiffWave-LibriSpeech t=3", "protect_foreign_seg_libri_t3.csv",
     "protect_foreign_bpda_libri_t3.csv"),
    ("DiffWave-SC09 t=1", "protect_foreign_seg_sc09_t1.csv",
     "protect_foreign_bpda_sc09_t1.csv"),
    ("DiffWave-SC09 t=3", "protect_foreign_seg_sc09_t3.csv",
     "protect_foreign_bpda_sc09_t3.csv"),
]
ARMS = [("constrained (fullnsT)", "fullnsT_tc_m30_seg", "fullnsT_tc_m30_seg_bpda"),
        ("unconstrained (full)", "full_tc_m30_seg", "full_tc_m30_seg_bpda")]


def margin_col(d):
    if "margin" not in d.columns:
        d = d.copy()
        d["margin"] = d.cos_after_purify - d.threshold
    return d


def speaker_of(d):
    return d.file.astype(str).str.extract(r"^spk(\d+)_")[0].astype(int)


def paired(exact_path, bpda_path, exact_tag, bpda_tag, enc):
    """per-speaker (exact - bpda) margin differences, inner-joined on speaker"""
    for p in (exact_path, bpda_path):
        if not os.path.isfile(p):
            return None
    de = margin_col(pd.read_csv(exact_path))
    db = margin_col(pd.read_csv(bpda_path))
    for d in (de, db):
        d["speaker"] = speaker_of(d)
    se = de[(de.encoder == enc) & (de.variant == exact_tag)]
    sb = db[(db.encoder == enc) & (db.variant == bpda_tag)]
    if len(se) == 0 or len(sb) == 0:
        return None
    a = se.groupby("speaker")["margin"].mean()
    b = sb.groupby("speaker")["margin"].mean()
    j = pd.concat([a.rename("exact"), b.rename("bpda")], axis=1).dropna()
    return (j["exact"] - j["bpda"]).to_numpy()


def pct_ci(dd, alpha, n_boot=4000, seed=0):
    """percentile bootstrap CI at level (1 - 2*alpha), the TOST-CI identity"""
    rng = np.random.default_rng(seed)
    n = len(dd)
    boots = dd[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(np.percentile(boots, 100 * alpha)),
            float(np.percentile(boots, 100 * (1 - alpha))))


def tost_t(dd, D):
    """Schuirmann's two one-sided t-tests; returns the larger (governing) p."""
    from scipy import stats
    n = len(dd)
    m, se = dd.mean(), dd.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < D else 1.0
    p_lo = stats.t.sf((m + D) / se, df=n - 1)     # H01: mu <= -D
    p_hi = stats.t.cdf((m - D) / se, df=n - 1)    # H02: mu >= +D
    return float(max(p_lo, p_hi))


def achieved_bound(dd, alpha):
    """smallest D at which equivalence would hold = max |CI endpoint|"""
    lo, hi = pct_ci(dd, alpha)
    return max(abs(lo), abs(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.01,
                    help="equivalence margin D; 0.01 is the PLAN_V3 "
                         "pre-specified value")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(R, "tost_bpda.csv"))
    a = ap.parse_args()
    D = a.margin

    print("=" * 110)
    print(f"RESULTS 30 AS AN EQUIVALENCE TEST -- TOST, margin D = {D:.3f}, "
          f"alpha = {a.alpha}")
    print("endpoint: paired per-speaker (exact - BPDA) margin. "
          "Equivalence iff the 90 % CI lies inside (-D, +D).")
    print("=" * 110)

    rows = []
    for arm_label, et, bt in ARMS:
        cells = []
        for ch_label, ef, bf in CHANNELS:
            for enc in ENC_ORDER:
                dd = paired(os.path.join(R, ef), os.path.join(R, bf),
                            et, bt, enc)
                if dd is None or len(dd) < 5:
                    continue
                lo, hi = pct_ci(dd, a.alpha)
                cells.append(dict(
                    arm=arm_label, channel=ch_label, encoder=enc, n=len(dd),
                    delta=float(dd.mean()), lo=lo, hi=hi,
                    p_tost=tost_t(dd, D),
                    achieved=achieved_bound(dd, a.alpha),
                    equivalent=bool(lo > -D and hi < D)))
        if not cells:
            print(f"\n  [{arm_label}] no data")
            continue
        for c, pa in zip(cells, holm([c["p_tost"] for c in cells])):
            c["p_tost_holm"] = pa
            c["equivalent_holm"] = bool(c["equivalent"] and pa < a.alpha)

        print(f"\n  {arm_label}     ({len(cells)} cells)")
        print(f"  {'channel':26s} {'encoder':22s} {'delta':>9s} "
              f"{'90% CI':>20s} {'p_TOST':>8s} {'p_holm':>8s} "
              f"{'achieved':>9s}  equiv")
        for c in cells:
            print(f"  {c['channel']:26s} {c['encoder']:22s} "
                  f"{c['delta']:+9.4f} [{c['lo']:+8.4f},{c['hi']:+8.4f}] "
                  f"{c['p_tost']:8.4f} {c['p_tost_holm']:8.4f} "
                  f"{c['achieved']:9.4f}  "
                  f"{'YES' if c['equivalent_holm'] else 'no'}")
        k = sum(c["equivalent_holm"] for c in cells)
        worst = max(c["achieved"] for c in cells)
        print(f"  -> equivalence at D = {D:.3f} in {k} of {len(cells)} cells "
              f"(Holm-adjusted)")
        print(f"  -> ALL cells are equivalent at D = {worst:.4f} "
              f"(the achieved bound)")
        rows += cells

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    print("\n" + "=" * 110)
    print("WHAT MAY BE WRITTEN")
    print("=" * 110)
    if len(df):
        k = int(df.equivalent_holm.sum())
        worst = float(df.achieved.max())
        print(f"  {k} of {len(df)} cells equivalent at the pre-specified "
              f"D = {D:.3f}; every cell equivalent at D = {worst:.4f}.")
        if k == len(df):
            print("\n  Write: \"BPDA and exact gradients are statistically")
            print(f"  EQUIVALENT within {D:.2f} margin units (TOST, alpha ="
                  f" {a.alpha}, Holm over")
            print("  the cells of each arm, speaker as the unit).\" The claim")
            print("  'the gradient estimator is not the lever' is now")
            print("  supported by a test of the thing it asserts, and the")
            print("  dev-clean run's use of BPDA is licensed.")
        else:
            print(f"\n  Write: \"equivalent at D = {worst:.3f}, which is the")
            print("  bound the data support; equivalence at the pre-specified")
            print(f"  {D:.2f} fails in {len(df) - k} cell(s).\" Name them.")
            print("  Do NOT write 'equivalent' unqualified, and do NOT move")
            print("  the margin to make the verdict pass -- the achieved")
            print("  bound is the honest statement.")
        print("\n  Never write: \"no significant difference, therefore the")
        print("  same\". That was the defect this section fixes.")
        print("\n  Note (RESULTS 50): exact and BPDA arms are two SEPARATE")
        print("  crafting runs, so each paired difference carries a run-to-run")
        print("  term of up to ~0.03 per utterance. That widens these CIs and")
        print("  makes equivalence harder, not easier -- the verdict is")
        print("  conservative.")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
