"""
experiments/analyse_alpha_pairs.py
==================================
Council #8, the C1 half: **all** pairwise α comparisons under one Holm family.

WHAT §32b DID, AND WHY IT IS NOT ENOUGH
---------------------------------------
§32b reports five pairwise comparisons of α between directions, each with its
own p-value:

    WavLM > d-vector, ECAPA > d-vector, ECAPA > x-vector,
    WavLM > ECAPA, random > WavLM

With 5 directions there are **10** pairs. Reporting 5 of them, selected after
seeing the means, and quoting unadjusted p-values, is two problems at once:
the selection is post hoc, and the multiplicity is unaccounted. `FRAME_v6` §2
already restricts C1 to "a two-group separation plus d-vector nsr > 1, never a
four-way ranking" partly for this reason. This script computes all ten under a
single Holm family so the claim can be stated at whatever resolution the data
actually supports, rather than at the resolution the first five p-values
suggested.

ESTIMATOR
---------
  unit          speaker (paired: every direction is measured on the same
                utterance of the same speaker through the same purifier seeds)
  statistic     mean paired difference in α, direction A minus direction B
  interval      BCa bootstrap 95 % CI, 2000 resamples
  p             Wilcoxon signed-rank, two-sided
  family        ALL pairs among the directions present, one Holm family per
                timestep. Not per encoder, not per hand-picked subset.
  significant   CI excluding 0 AND Holm p < 0.05

A Friedman test across directions is reported first as the omnibus: if it does
not reject, the pairwise table is descriptive and is labelled that way.

Run:
  python experiments/analyse_alpha_pairs.py
  python experiments/analyse_alpha_pairs.py --in experiments/results/encoder_subspace_xsess_fe.csv
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.analyse_matched_control import bca_ci, wilcoxon_p, holm

R = os.path.join(PROJ, "experiments", "results")


def friedman(piv):
    """Friedman chi-square across the columns of a complete speaker x direction
    table. Returns (stat, p, k, n) or (nan, nan, k, n) without scipy."""
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        return float("nan"), float("nan"), piv.shape[1], piv.shape[0]
    cols = [piv[c].to_numpy() for c in piv.columns]
    try:
        st, p = friedmanchisquare(*cols)
    except ValueError:
        return float("nan"), float("nan"), piv.shape[1], piv.shape[0]
    return float(st), float(p), piv.shape[1], piv.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(R, "encoder_subspace_matched.csv"))
    ap.add_argument("--t", default="50,200")
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--include-controls", action="store_true",
                    dest="include_controls",
                    help="also pair the band_matched / pca_span controls; off "
                         "by default because §37 already tests each encoder "
                         "against its own control and those are a different "
                         "family")
    ap.add_argument("--out", default=os.path.join(R, "alpha_pairs.csv"))
    a = ap.parse_args()

    if not os.path.isfile(a.inp):
        raise SystemExit(f"not found: {a.inp}")
    df = pd.read_csv(a.inp)
    rows = []
    for t in [int(v) for v in a.t.split(",")]:
        col = f"alpha_t{t}"
        if col not in df.columns:
            print(f"[pairs] {col} absent, skipping t={t}")
            continue
        piv = df.pivot_table(index="speaker", columns="direction", values=col)
        dirs = [c for c in piv.columns
                if a.include_controls
                or not str(c).startswith(("band_matched", "pca_span"))]
        piv = piv[dirs].dropna()
        pairs = list(itertools.combinations(sorted(dirs), 2))

        st, pf, k, n = friedman(piv)
        print("\n" + "=" * 96)
        print(f"ALL PAIRWISE alpha COMPARISONS, t = {t}   "
              f"({os.path.basename(a.inp)})")
        print(f"n = {n} speakers, {k} directions, {len(pairs)} pairs, "
              f"ONE Holm family")
        if pf == pf:
            verdict = "rejects" if pf < a.alpha else "does NOT reject"
            print(f"Friedman omnibus: chi2 = {st:.1f}, p = {pf:.2e} -- "
                  f"{verdict} equality of all {k} directions")
            if pf >= a.alpha:
                print("  -> the pairwise table below is DESCRIPTIVE only.")
        print("=" * 96)

        cells, ps = [], []
        for x, y in pairs:
            d = (piv[x] - piv[y]).to_numpy()
            lo, hi, _, _ = bca_ci(d, alpha=a.alpha, n_boot=a.n_boot)
            p = wilcoxon_p(d)
            cells.append(dict(t=t, a=x, b=y, n=len(d), delta=float(d.mean()),
                              lo=lo, hi=hi, p=p,
                              higher=int((d > 0).sum())))
            ps.append(p)
        for c, padj in zip(cells, holm(ps)):
            c["p_holm"] = padj
            c["sig"] = bool((c["hi"] < 0 or c["lo"] > 0) and padj < a.alpha)
        cells.sort(key=lambda c: -abs(c["delta"]))

        w = max(len(c["a"]) for c in cells)
        print(f"  {'direction A':{w}s}   {'direction B':{w}s} {'A-B':>8s} "
              f"{'95% CI':>19s} {'A>B':>7s} {'p_holm':>9s}  sig")
        for c in cells:
            print(f"  {c['a']:{w}s} - {c['b']:{w}s} {c['delta']:+8.4f} "
                  f"[{c['lo']:+8.4f},{c['hi']:+8.4f}] "
                  f"{c['higher']:3d}/{c['n']:<3d} {c['p_holm']:9.2e}  "
                  f"{'*' if c['sig'] else ''}")
        rows.extend(cells)

        sig = [c for c in cells if c["sig"]]
        print(f"\n  {len(sig)} of {len(cells)} pairs separate under Holm.")
        ns = [c for c in cells if not c["sig"]]
        if ns:
            print("  NOT separated: " + "; ".join(
                f"{c['a'].split()[0][:9]} vs {c['b'].split()[0][:9]}"
                for c in ns))

    if rows:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\nsaved {a.out}")
        print("\nFRAME_v6 §4 forbids quoting this as a four-way ranking. Use it")
        print("to say exactly which directions separate and which do not, at")
        print("the resolution the Holm column supports.")


if __name__ == "__main__":
    main()
