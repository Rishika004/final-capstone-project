"""experiments/analyse_transfer.py -- RESULTS section 33 on the corrected base.

Produces every number in section 33 from `foreign_tc_*.csv` and
`survival_tc_v7.csv`, so the section is regenerable rather than transcribed.

WHY A SCRIPT
------------
Council #2: the lead result was quoted on the old base (n = 20) while these
files sat on disk unanalysed, and the old numbers were hard-coded in the table
generators. Council #8: retention was reported above 100 % with no interval, and
the Wilcoxon p was quoted as a strength when it was at its floor.

Both are fixed here:

  retention  = excess_foreign / excess_ours, a ratio of ratios of means, so its
               CI comes from a speaker-level bootstrap on the log scale (BCa,
               jackknife acceleration), not from any closed form. Reported as a
               percentage with that interval. An interval spanning 100 % means
               "indistinguishable from full retention", which is the honest
               reading of the two t=3 cells.

  separation = the count of speakers whose pass-through alpha exceeds their own
               random alpha, with an exact sign test. When every speaker
               separates, the Wilcoxon p sits at 2/2^n and carries no
               information beyond the count, so the count is what is reported.

Run:
  python experiments/analyse_transfer.py
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
CELLS = (("foreign_tc_libri_t1.csv", "DiffWave-LibriSpeech", 1),
         ("foreign_tc_libri_t3.csv", "DiffWave-LibriSpeech", 3),
         ("foreign_tc_sc09_t1.csv", "DiffWave-SC09", 1),
         ("foreign_tc_sc09_t3.csv", "DiffWave-SC09", 3))


def _ppf(p):
    from statistics import NormalDist
    return NormalDist().inv_cdf(float(np.clip(p, 1e-12, 1 - 1e-12)))


def _cdf(z):
    from statistics import NormalDist
    return NormalDist().cdf(float(z))


def bca_statistic(stat, n, alpha=0.05, n_boot=2000, seed=20261005):
    """BCa CI for any statistic computed from a speaker INDEX array.

    `stat(idx)` must accept an array of row indices (with repeats) and return a
    scalar. This is the general form needed for a ratio of ratios of means,
    which is not the mean of anything and so cannot use the simple BCa path.
    """
    theta = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if boots.size < 50:
        return float("nan"), float("nan"), theta

    frac = float((boots < theta).mean())
    z0 = 0.0 if frac <= 0 or frac >= 1 else _ppf(frac)
    jack = np.array([stat(np.delete(np.arange(n), i)) for i in range(n)])
    dev = jack.mean() - jack
    den = 6.0 * (dev ** 2).sum() ** 1.5
    a = float((dev ** 3).sum() / den) if den > 0 else 0.0

    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = _ppf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(boots, 100 * _cdf(adj))))
    return out[0], out[1], theta


def sign_test(k, n):
    """Exact two-sided sign test for k of n in one direction."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, 0.5).pvalue)
    except ImportError:
        from math import comb
        m = min(k, n - k)
        return float(min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n))


def load_cell(fname):
    d = read_csv(os.path.join(RES, fname))
    p = d[d.direction == "pass_through"].set_index("speaker").sort_index()
    r = d[d.direction == "random"].set_index("speaker").sort_index()
    common = p.index.intersection(r.index)
    return p.loc[common], r.loc[common]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    print("=" * 78)
    print("RESULTS 33 -- purifier transfer on the CORRECTED base "
          "(test-clean / v7 / cross-session)")
    print("=" * 78)
    print()
    print(f"  {'cell':24s} {'t':>2s} {'a_pass':>7s} {'a_rand':>7s} {'sep':>7s} "
          f"{'sign p':>9s} {'reten':>7s} {'95% CI':>18s}")

    summary = []
    for fname, label, t in CELLS:
        path = os.path.join(RES, fname)
        if not os.path.isfile(path):
            print(f"  {label:24s} {t:2d}  MISSING ({fname})")
            continue
        p, r = load_cell(fname)
        n = len(p)
        af, ar = p.alpha_foreign.to_numpy(), r.alpha_foreign.to_numpy()
        ao, aro = p.alpha_ours.to_numpy(), r.alpha_ours.to_numpy()
        av, arv = p.alpha_vocoder.to_numpy(), r.alpha_vocoder.to_numpy()

        sep = int((af > ar).sum())
        sp = sign_test(sep, n)

        def retention(idx):
            voc = av[idx].mean() / arv[idx].mean()
            ex_f = (af[idx].mean() / ar[idx].mean()) / voc
            ex_o = (ao[idx].mean() / aro[idx].mean()) / voc
            return np.log(ex_f / ex_o)          # log scale, then exponentiate

        lo, hi, theta = bca_statistic(retention, n, a.alpha, a.n_boot)
        ret, rlo, rhi = (100 * np.exp(v) for v in (theta, lo, hi))
        voc = av.mean() / arv.mean()
        summary.append(dict(label=label, t=t, n=n, a_pass=af.mean(),
                            a_rand=ar.mean(), sep=sep, p=sp, ret=ret,
                            lo=rlo, hi=rhi, voc=voc))
        print(f"  {label:24s} {t:2d} {af.mean():7.3f} {ar.mean():7.3f} "
              f"{sep:3d}/{n:<3d} {sp:9.2e} {ret:6.0f}% "
              f"[{rlo:6.0f}%,{rhi:6.0f}%]")

    if not summary:
        raise SystemExit("no transfer cells on disk")

    vocs = [s["voc"] for s in summary]
    print(f"\n  vocoder floor (mel + Griffin-Lim round trip alone): "
          f"{np.mean(vocs):.2f}x  [{min(vocs):.2f}, {max(vocs):.2f}]")
    print("  `excess` divides this floor out; `retention` is the foreign excess")
    print("  as a fraction of our own purifier's.")

    spans = [s for s in summary if s["lo"] <= 100 <= s["hi"]]
    print(f"\n  cells whose retention CI spans 100 %: {len(spans)} of {len(summary)}"
          + (" -- " + ", ".join(f"{s['label'].split('-')[-1]} t={s['t']}"
                                for s in spans) if spans else ""))
    print("  For those, say 'indistinguishable from full retention', NOT")
    print("  'exceeds 100 %': a point estimate above 100 with an interval that")
    print("  includes it is not evidence of amplification (council #8).")

    allsep = [s for s in summary if s["sep"] == s["n"]]
    if allsep:
        n0 = allsep[0]["n"]
        print(f"\n  {len(allsep)} of {len(summary)} cells separate on EVERY speaker "
              f"({n0}/{n0}). The Wilcoxon p is then pinned at its floor 2/2^{n0} "
              f"= {2 / 2 ** n0:.1e}")
        print("  and restates the count; quote the count and the sign test, not the p.")

    # ----------------------------------------------------- survival reference
    sf = os.path.join(RES, "survival_tc_v7.csv")
    if os.path.isfile(sf):
        d = read_csv(sf)
        print("\n" + "=" * 78)
        print(f"survival through OUR purifier v7, n = {d.speaker.nunique()} "
              f"(survival_tc_v7.csv)")
        print("=" * 78)
        print(f"  {'kind':26s} " + "".join(f"{f'a t={t}':>9s}{f'nsr t={t}':>10s}"
                                           for t in (50, 200)))
        has_ctrl = False
        for kind in ("null", "null_t200", "random", "range",
                     "band_matched:null", "band_matched:null_t200", "pca_span"):
            s = d[d.kind == kind]
            if not len(s):
                continue
            if kind.startswith(("band_matched", "pca_span")):
                has_ctrl = True
            print(f"  {kind:26s} " + "".join(
                f"{s[f'alpha_t{t}'].mean():9.3f}{s[f'nsr_t{t}'].mean():10.2f}"
                for t in (50, 200)))
        piv = d.pivot_table(index="speaker", columns="kind", values="alpha_t50")
        if "null" in piv and "random" in piv:
            k = int((piv["null"] > piv["random"]).sum())
            print(f"\n  pass-through beats white-noise random on {k}/{len(piv)} "
                  f"speakers at t=50 (sign test p = {sign_test(k, len(piv)):.1e})")
        if not has_ctrl:
            print("\n  MATCHED CONTROL PENDING (council #3): the only baseline in this")
            print("  file is isotropic white noise, which this purifier was trained to")
            print("  remove. Re-run exp_nullspace_survival.py without --no-controls.")


if __name__ == "__main__":
    main()
