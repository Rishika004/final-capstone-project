"""experiments/analyse_bpda_null.py -- the empirical noise floor for a crafting-loop edit.

PLAN_V3 2b: BPDA-vs-exact (RESULTS section 30) is a measured change to the
crafting loop with no real effect, on the same base, seeds, budget and
thresholds. Its paired per-speaker spread is therefore an EMPIRICAL NULL: the
amount a gate can see move when nothing has actually changed. Every future gate
should state itself as a multiple of it.

WHY THIS SCRIPT EXISTS RATHER THAN A CONSTANT
---------------------------------------------
PLAN_V3 2b and the required gate wording in PLAN_V3 section 4 both quote a
single number, 0.0096, as "the crafting-change component". That number is real
but it is the SMALLEST of the 18 (arm x encoder x eval_t) cells -- the
constrained arm, d-vector, t=50. Across all 18 cells the paired SD runs from
0.0086 to 0.0557, median 0.0247. A gate calibrated on 0.0096 but read in
another cell is calibrated 2.6x too tight at the median and 5.8x too tight at
the worst, and would fire on noise.

The companion figure PLAN_V3 2b quotes for the "worst margin", 0.0102, does not
reproduce under any natural definition: the worst-encoder-per-speaker margin has
SD 0.0192 (constrained arm, all t) to 0.0451 (unconstrained, t=50).

So the floor is a TABLE, not a constant, and a gate must take the number from
the cell it names. That is the same discipline PLAN_V3 section 4 already imposes
on the endpoint ("name the cell, column, arm and estimator"); it has to apply to
the noise floor too, or the gate is calibrated against a different cell than it
tests.

Run:
  python experiments/analyse_bpda_null.py
"""
import argparse
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from utils.csvio import read_csv

RES = os.path.join(PROJ, "experiments", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", default="main_table_tc_m30_seg.csv")
    ap.add_argument("--bpda", default="main_table_tc_m30_seg_bpda.csv")
    a = ap.parse_args()

    ex = read_csv(os.path.join(RES, a.exact))
    bp = read_csv(os.path.join(RES, a.bpda)).copy()
    # The BPDA run tags its arms with a _bpda suffix; strip it so the arms pair.
    bp["variant"] = bp.variant.str.replace("_bpda", "", regex=False)
    key = ["variant", "file", "encoder", "eval_t"]
    m = ex.merge(bp, on=key, suffixes=("_ex", "_bp"), validate="one_to_one")
    m["margin_ex"] = m.cos_after_purify_ex - m.threshold_ex
    m["margin_bp"] = m.cos_after_purify_bp - m.threshold_bp
    m["d"] = m.margin_bp - m.margin_ex

    print("=" * 78)
    print("EMPIRICAL NULL for a crafting-loop edit (BPDA vs exact gradients)")
    print("=" * 78)
    print(f"{a.exact} vs {a.bpda}")
    print("Same base, speakers, seeds, budget, thresholds. Only the gradient")
    print("estimator differs, and RESULTS 30 found it is not the delta. So this")
    print("spread is what a gate sees when nothing real has changed.")
    print(f"\npaired rows: {len(m)} | speakers: {m.file.nunique()}\n")

    print(f"  {'arm':22s} {'encoder':22s} {'t':>4s} {'mean d':>9s} "
          f"{'SD':>8s} {'n':>4s}")
    cells = []
    for (v, e, t), g in m.groupby(["variant", "encoder", "eval_t"]):
        s = g.d.std(ddof=1)
        cells.append(dict(arm=v, encoder=e, t=t, mean=g.d.mean(), sd=s, n=len(g)))
        print(f"  {v:22s} {e:22s} {t:4d} {g.d.mean():+9.4f} {s:8.4f} {len(g):4d}")

    sds = np.array([c["sd"] for c in cells])
    lo = min(cells, key=lambda c: c["sd"])
    hi = max(cells, key=lambda c: c["sd"])
    print(f"\n  across {len(cells)} cells: min {sds.min():.4f}  "
          f"median {np.median(sds):.4f}  max {sds.max():.4f}")
    print(f"  smallest: {lo['arm']} / {lo['encoder']} / t={lo['t']} "
          f"= {lo['sd']:.4f}")
    _p = m[m.variant.str.startswith("fullnsT")
           & (m.encoder == "Resemblyzer d-vector") & (m.eval_t == 50)]
    if len(_p):
        print(f"  PLAN_V3's 0.0096 is a different cell again: fullnsT / d-vector "
              f"/ t=50 = {_p.d.std(ddof=1):.4f}")
    print(f"  largest : {hi['arm']} / {hi['encoder']} / t={hi['t']} "
          f"= {hi['sd']:.4f}")
    print(f"  a gate calibrated on {lo['sd']:.4f} is "
          f"{np.median(sds) / lo['sd']:.1f}x too tight at the median cell and "
          f"{sds.max() / lo['sd']:.1f}x too tight at the worst.")

    # The constrained arm is the shipping one, so its row is the one most gates
    # will actually want; print it on its own.
    print("\n  SHIPPING ARM ONLY (fullnsT), which is what a gate normally names:")
    sub = m[m.variant.str.startswith("fullnsT")]
    for (e, t), g in sub.groupby(["encoder", "eval_t"]):
        print(f"    {e:22s} t={t:<4d} SD {g.d.std(ddof=1):.4f}")

    # Worst-encoder-per-speaker margin, the other quantity PLAN_V3 2b names.
    print("\n  WORST-ENCODER margin per speaker (the other 2b quantity):")
    for arm in sorted(m.variant.unique()):
        s = m[m.variant == arm]
        w = s.groupby(["file", "eval_t"]).apply(
            lambda g: g.margin_bp.min() - g.margin_ex.min(), include_groups=False)
        print(f"    {arm:22s} SD {w.std(ddof=1):.4f} (n={len(w)})")
    print("    PLAN_V3 2b quotes 0.0102 here; no definition tried reproduces it.")

    print("\n" + "-" * 78)
    print("HOW TO USE THIS")
    print("  Take the floor from the cell the gate names, not the minimum.")
    print("  Combine with the crafting-SEED component (RESULTS 31, ~0.0127) in")
    print("  quadrature when a gate spans seeds as well as a code change.")


if __name__ == "__main__":
    main()
