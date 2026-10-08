"""experiments/analyse_matched_control.py -- the council #3 gate, decided properly.

Reads `encoder_subspace.csv` (with the matched controls from PLAN_V3 W1) and
answers the one question C1 depends on:

    does each encoder's identity direction survive purification WORSE than a
    random direction carrying that encoder direction's own spectral envelope?

Estimator, fixed by PLAN_V3 section 4 before the run:
  unit            = speaker (paired; every direction is measured on the same
                    utterance of the same speaker through the same seeds)
  statistic       = mean paired difference  alpha(g_e) - alpha(control_e)
  interval        = BCa bootstrap 95 % CI, 2000 resamples (bias-corrected and
                    accelerated: the paired differences are skewed, so a
                    percentile interval would be mis-centred)
  p               = Wilcoxon signed-rank, two-sided
  multiplicity    = Holm across the encoders in the table

A bare mean is not the gate. PLAN_V3 section 4: the d-vector margin spans 0.047
across legitimate analysis choices, so a point estimate alone decides nothing.

CALIBRATION, measured before the run (simulation at n = 29, per-speaker SD 0.05,
4 encoders, 400 global-null replicates):

    BCa upper bound < 0 alone .............. 0.035   (anti-conservative at n=29)
    the gate as implemented, CI AND Holm ... 0.030 family-wise, 0.007 per encoder

So a CONFIRMED verdict is reported only on the conjunction, never on the CI
alone. The CI is quoted everywhere because its width is the informative part.

Run:
  python experiments/analyse_matched_control.py
  python experiments/analyse_matched_control.py --t 200 --control pca_span
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

DEFAULT_IN = os.path.join(PROJ, "experiments", "results", "encoder_subspace.csv")


def _ppf(p):
    """Standard-normal quantile without a scipy dependency at import time."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(float(np.clip(p, 1e-12, 1 - 1e-12)))


def _cdf(z):
    from statistics import NormalDist
    return NormalDist().cdf(float(z))


def bca_ci(x, alpha=0.05, n_boot=2000, seed=20261005):
    """BCa bootstrap CI for the mean of a 1-D sample.

    Returns (lo, hi, z0, a). Falls back to the percentile interval when the
    acceleration is undegenerate-undefined (all values identical).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    theta = x.mean()
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)

    # bias correction: where theta sits in the bootstrap distribution
    frac = float((boots < theta).mean())
    if frac <= 0 or frac >= 1:
        z0 = 0.0
    else:
        z0 = _ppf(frac)

    # acceleration from the jackknife
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    dev = jack.mean() - jack
    denom = 6.0 * (dev ** 2).sum() ** 1.5
    a = float((dev ** 3).sum() / denom) if denom > 0 else 0.0

    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = _ppf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(boots, 100 * _cdf(adj))))
    return out[0], out[1], z0, a


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p; exact for small n via scipy."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals):
    """Holm-adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    adj = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = order.size
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--t", default="50", help="purifier strength column to read")
    ap.add_argument("--control", default="band_matched",
                    choices=["band_matched", "pca_span", "random"],
                    help="band_matched = per-encoder envelope-matched (the gate); "
                         "pca_span = clean-speech principal subspace; "
                         "random = the old isotropic white-noise baseline")
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    if not os.path.isfile(a.inp):
        raise SystemExit(f"not found: {a.inp}")
    df = read_csv(a.inp)
    col = f"alpha_t{a.t}"
    if col not in df.columns:
        raise SystemExit(f"{col} not in {a.inp}; have "
                         f"{[c for c in df.columns if c.startswith('alpha')]}")

    piv = df.pivot_table(index="speaker", columns="direction", values=col)
    encoders = [d for d in piv.columns
                if not str(d).startswith(("band_matched", "pca_span"))
                and d != "random"]
    if not encoders:
        raise SystemExit("no encoder directions found")

    # RESULTS 58: if the file carries the 57f instrument gate, name the cells
    # that failed it. They stay in the pre-registered statistic; they are
    # reported beside it so a reader can see what the mean contains.
    if "gate_passed" in df:
        bad = df[(df.gate_passed.astype(str) == "False") & (~df.is_control)]
        if len(bad):
            print(f"[matched] instrument gate (57f/58) FAILED for {len(bad)} cell(s): "
                  + "; ".join(f"spk{s_}: {sorted(g_.direction)}"
                              for s_, g_ in bad.groupby("speaker")))
            print("[matched] no verdict may be read from those cells on their own.")
    has_matched = any(str(c).startswith("band_matched") for c in piv.columns)
    if a.control == "band_matched" and not has_matched:
        raise SystemExit(
            f"{os.path.basename(a.inp)} has no band_matched control -- it predates\n"
            "PLAN_V3 W1 #3. Re-run exp_encoder_subspace.py without --no-controls,\n"
            "or pass --control random to read the old white-noise baseline.")

    print("=" * 78)
    print(f"COUNCIL #3 GATE   {os.path.basename(a.inp)}   {col}   "
          f"control = {a.control}")
    print("=" * 78)
    shas = sorted(set(df["code_sha"].dropna().astype(str))) if "code_sha" in df else []
    if shas:
        print(f"code_sha: {', '.join(shas)}")
    if a.control == "pca_span" and "pca_k" in df.columns:
        info = df[df.direction == "pca_span"][["pca_k", "pca_subset", "pca_var"]]
        if len(info):
            r0 = info.iloc[0]
            print(f"pca basis: k={r0['pca_k']} from {r0['pca_subset']} "
                  f"(var explained {r0['pca_var']})")
    print(f"n = {len(piv)} speakers | estimator = mean paired difference, "
          f"BCa {int((1 - a.alpha) * 100)} % CI ({a.n_boot} resamples), "
          f"Wilcoxon signed-rank, Holm over {len(encoders)} encoders")
    print()

    rows, ps = [], []
    for e in sorted(encoders):
        ctrl = f"band_matched:{e}" if a.control == "band_matched" else a.control
        if ctrl not in piv.columns:
            print(f"  (skip {e}: no control column {ctrl})")
            continue
        sub = piv[[e, ctrl]].dropna()
        d = (sub[e] - sub[ctrl]).to_numpy()
        lo, hi, z0, acc = bca_ci(d, alpha=a.alpha, n_boot=a.n_boot)
        p = wilcoxon_p(d)
        ps.append(p)
        rows.append(dict(encoder=e, n=len(d), alpha=sub[e].mean(),
                         control=sub[ctrl].mean(), delta=d.mean(),
                         lo=lo, hi=hi, below=int((d < 0).sum()), p=p,
                         z0=z0, acc=acc))

    if not rows:
        raise SystemExit("nothing to test")
    for r, padj in zip(rows, holm(ps)):
        r["p_holm"] = padj

    w = max(len(r["encoder"]) for r in rows)
    print(f"  {'encoder':{w}s} {'alpha':>7s} {'ctrl':>7s} {'delta':>8s} "
          f"{'95% CI':>18s} {'below':>7s} {'p':>9s} {'p_holm':>9s}")
    for r in rows:
        print(f"  {r['encoder']:{w}s} {r['alpha']:+7.3f} {r['control']:+7.3f} "
              f"{r['delta']:+8.4f} [{r['lo']:+7.4f},{r['hi']:+7.4f}] "
              f"{r['below']:3d}/{r['n']:<3d} {r['p']:9.2e} {r['p_holm']:9.2e}")

    # ------------------------------------------------- the pre-registered read
    print("\n" + "-" * 78)
    sig = [r for r in rows if r["hi"] < 0 and r["p_holm"] < a.alpha]
    null = [r for r in rows if not (r["hi"] < 0 and r["p_holm"] < a.alpha)]
    print(f"VERDICT  ({len(sig)}/{len(rows)} encoders survive worse than their "
          f"matched control,\n         CI upper bound < 0 and Holm p < {a.alpha})")
    for r in sig:
        print(f"  CONFIRMED  {r['encoder']}: {r['delta']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}]")
    for r in null:
        why = "CI includes 0" if r["hi"] >= 0 else "Holm p not significant"
        print(f"  NOT SHOWN  {r['encoder']}: {r['delta']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] -- {why}")

    # Does the ORDERING survive? C1's claim is an ordering, not just a sign.
    print("\nORDERING (alpha, and alpha minus its matched control):")
    order_raw = sorted(rows, key=lambda r: -r["alpha"])
    order_adj = sorted(rows, key=lambda r: -r["delta"])
    print("  by alpha  : " + " > ".join(f"{r['encoder'].split()[0][:9]}"
                                        for r in order_raw))
    print("  by delta  : " + " > ".join(f"{r['encoder'].split()[0][:9]}"
                                        for r in order_adj))
    if [r["encoder"] for r in order_raw] == [r["encoder"] for r in order_adj]:
        print("  -> the ordering is NOT an artefact of per-encoder spectral envelope.")
    else:
        print("  -> the ordering CHANGES once each encoder is read against its own\n"
              "     envelope. C1's ordering is partly spectral; say so explicitly.")

    print("\nRead against the pre-registered branches in core/matched_random.py.")
    print("If this is a null, FRAME_v6 section 6 pre-holds the branch: C1 becomes a")
    print("spectral statement, C3 leads, C2 becomes the mechanism. Do not re-run")
    print("looking for a cell where it clears.")


if __name__ == "__main__":
    main()
