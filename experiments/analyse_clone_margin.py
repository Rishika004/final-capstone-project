"""experiments/analyse_clone_margin.py -- the cloning headline, unfiltered.  (PLAN_V3 2a)

WHAT THIS REPLACES
------------------
Every cloning number in the paper has been a BINARY rate (`clone_succeeds`)
computed on a PRE-FILTERED speaker set: speakers were kept only when the
unprotected clone succeeded, so the defender's endpoint was conditioned on a
noisy draw of the defender's own outcome. Consequences already measured:

  * council #4: 56/174 (32 %) of unprotected cells flip between identical runs,
    and the keep-set size wanders 9 / 6 / 7 -- a winner's curse favouring us.
  * RESULTS 31b: one cell reads 0 % / 33 % / 0 % across three crafting seeds.
  * RESULTS 31c: cells with n < 12 must not be reported alone.

The same CSVs carry a CONTINUOUS `margin` column (clone_sim - threshold) beside
the binary. Paired per (speaker, purifier), protected minus unprotected, it needs
no pre-filter at all: every speaker contributes, whatever the unprotected clone
did. All three defects above are artefacts of binarising, so all three go away.

THE UNIT-OF-ANALYSIS CAVEAT (not in PLAN_V3 2a)
-----------------------------------------------
PLAN_V3 2a specifies "paired per (speaker, purifier) ... n = 58". Those 58 rows
are 29 speakers crossed with 2 purifier conditions, so they are NOT 58
independent units: the two rows of a speaker share that speaker's voice,
reference utterance and threshold. Treating them as independent understates the
SE. This script therefore reports BOTH:

  n = 58  pooled rows, exactly the PLAN_V3 2a endpoint, for continuity;
  n = 29  speaker means over the purifier conditions -- the clustered unit, and
          the one quoted as primary, because its SE is honest.

The two agree on sign and significance here; where they disagree, the n = 29
figure is the one to report.

Run:
  python experiments/analyse_clone_margin.py
  python experiments/analyse_clone_margin.py --files cloning_tc_m30_deploy_seg
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from utils.csvio import read_csv
from experiments.analyse_matched_control import bca_ci, wilcoxon_p, holm

RES = os.path.join(PROJ, "experiments", "results")
DEFAULT_FILES = "cloning_tc_m30_seg_s7,cloning_tc_m30_seg_s99"


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI; (0,0) when n == 0."""
    if n == 0:
        return 0.0, 0.0
    try:
        from scipy.stats import beta
    except ImportError:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5).pvalue)
    except ImportError:
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        return float(min(1.0, 2 * tail))


def pair(df):
    """Protected minus unprotected on the (speaker, purifier, encoder) key."""
    prot = df[df.variant != "unprotected"]
    unp = df[df.variant == "unprotected"]
    if prot.empty or unp.empty:
        raise SystemExit("need both a protected arm and the unprotected arm")
    key = ["speaker", "purifier", "encoder"]
    m = prot.merge(unp, on=key, suffixes=("_p", "_u"), validate="one_to_one")
    m["delta"] = m["margin_p"] - m["margin_u"]
    return m


def describe(d, label, alpha=0.05):
    d = np.asarray(d, dtype=float)
    lo, hi, _, _ = bca_ci(d, alpha=alpha)
    return dict(unit=label, n=len(d), mean=d.mean(), sd=d.std(ddof=1),
                se=d.std(ddof=1) / np.sqrt(len(d)), lo=lo, hi=hi,
                p=wilcoxon_p(d), neg=int((d < 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=DEFAULT_FILES,
                    help="comma list of result CSV basenames (no .csv)")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    names = [f.strip() for f in a.files.split(",") if f.strip()]

    per_seed = {}
    for name in names:
        path = os.path.join(RES, f"{name}.csv")
        if not os.path.isfile(path):
            print(f"  (skip {name}: not on disk)")
            continue
        m = pair(read_csv(path))
        per_seed[name] = m

        print("=" * 78)
        print(f"PAIRED CONTINUOUS CLONE MARGIN -- {name}")
        print("=" * 78)
        print(f"arm: {sorted(set(m.variant_p))[0]} | purifiers: "
              f"{sorted(set(m.purifier))} | speakers: {m.speaker.nunique()} | "
              f"UNFILTERED (every speaker contributes)")
        print()

        rows, ps = [], []
        for enc in sorted(m.encoder.unique()):
            g = m[m.encoder == enc]
            pooled = describe(g.delta.to_numpy(), "n=58 pooled", a.alpha)
            byspk = g.groupby("speaker").delta.mean().to_numpy()
            clust = describe(byspk, "n=29 speaker", a.alpha)
            rows.append((enc, pooled, clust))
            ps.append(clust["p"])
        padj = holm(ps)

        print(f"  {'encoder':22s} {'unit':13s} {'mean':>8s} {'SD':>7s} {'SE':>7s} "
              f"{'95% CI':>19s} {'neg':>7s} {'t':>7s} {'p':>9s}")
        for (enc, pooled, clust), pa in zip(rows, padj):
            for r in (pooled, clust):
                t = r["mean"] / r["se"] if r["se"] else float("nan")
                tag = f"{r['p']:9.2e}" if r is pooled else f"{pa:9.2e}"
                print(f"  {enc[:22]:22s} {r['unit']:13s} {r['mean']:+8.4f} "
                      f"{r['sd']:7.4f} {r['se']:7.4f} "
                      f"[{r['lo']:+8.4f},{r['hi']:+8.4f}] "
                      f"{r['neg']:3d}/{r['n']:<3d} {t:+7.2f} {tag}")
        print("  (p on the n=29 line is Holm-corrected across encoders; the n=58\n"
              "   line is the PLAN_V3 2a endpoint, kept for continuity)")

        # ---------------------------------------- per purifier, so the cell is named
        print("\n  per purifier (speaker-level, n=29 each):")
        for purf in sorted(m.purifier.unique()):
            parts = []
            for enc in sorted(m.encoder.unique()):
                g = m[(m.encoder == enc) & (m.purifier == purf)]
                parts.append(f"{enc.split()[0][:9]}={g.delta.mean():+.4f}")
            print(f"    {purf:10s} " + "  ".join(parts))

        # ------------------------- the binary endpoint, for the power comparison
        print("\n  BINARY endpoint on the same rows (what the paper used to quote):")
        print(f"    {'encoder':22s} {'purifier':10s} {'unprot':>12s} {'prot':>12s} "
              f"{'McNemar p':>10s}  {'keep-set':>9s}")
        for enc in sorted(m.encoder.unique()):
            for purf in sorted(m.purifier.unique()):
                g = m[(m.encoder == enc) & (m.purifier == purf)]
                su, sp = g.clone_succeeds_u.astype(bool), g.clone_succeeds_p.astype(bool)
                n = len(g)
                lo_u, hi_u = clopper_pearson(int(su.sum()), n)
                lo_p, hi_p = clopper_pearson(int(sp.sum()), n)
                b = int((su & ~sp).sum())
                c = int((~su & sp).sum())
                print(f"    {enc[:22]:22s} {purf:10s} "
                      f"{su.sum():2d}/{n:<2d} [{lo_u:.2f},{hi_u:.2f}] "
                      f"{sp.sum():2d}/{n:<2d} [{lo_p:.2f},{hi_p:.2f}] "
                      f"{mcnemar_exact(b, c):10.2e}  {int(su.sum()):3d} kept")
        print("    keep-set = speakers the OLD pre-filtered analysis would have kept;\n"
              "    the continuous endpoint above uses all 29 regardless.")
        print()

    # ------------------------------------------------- seed stability
    if len(per_seed) > 1:
        print("=" * 78)
        print("SEED STABILITY of the continuous endpoint (speaker-level means)")
        print("=" * 78)
        encs = sorted(set().union(*(set(m.encoder.unique()) for m in per_seed.values())))
        print(f"  {'encoder':22s} " + "".join(f"{n.split('_')[-1]:>12s}"
                                              for n in per_seed) + f"{'range':>9s}")
        for enc in encs:
            vals = []
            for m in per_seed.values():
                g = m[m.encoder == enc]
                vals.append(g.groupby("speaker").delta.mean().mean() if len(g) else np.nan)
            rng = np.nanmax(vals) - np.nanmin(vals)
            print(f"  {enc[:22]:22s} " + "".join(f"{v:+12.4f}" for v in vals)
                  + f"{rng:9.4f}")
        print("\n  Compare the range column with the effect sizes above: the effect is")
        print("  an order of magnitude larger than the crafting-seed wobble, which is")
        print("  what RESULTS 31 found for continuous quantities and NOT for rates.")
        print("  The BPDA null (PLAN_V3 2b) gives the crafting-EDIT noise floor:")
        print("  per-speaker SD 0.0096 on the d-vector margin, 0.0102 on the worst.")


if __name__ == "__main__":
    main()
