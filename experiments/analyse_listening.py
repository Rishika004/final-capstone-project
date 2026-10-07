"""
experiments/analyse_listening.py
================================
Turn returned listening-test CSVs into the two tables the paper needs, with the
screener applied and the panel's composition stated rather than assumed.

INPUT
-----
Every file in `listening_test/responses/*.csv`, as exported by the page
(`experiments/make_listening_page.py`). Each row is one rating:

    listener, seed, block, stim_id, rating, order, screen_pass,
    anchor_hi_mean, anchor_lo_mean, headphones, naive, build_seed

The page is blind: it never knows which condition a `stim_id` is. Unblinding
happens here, by joining against `stimuli_key.csv`. That is the whole point of
the split -- the ratings cannot be influenced by the condition because nothing
in the browser knows it.

THE PANEL IS A CONVENIENCE SAMPLE, AND THE OUTPUT SAYS SO
---------------------------------------------------------
Council #6 asked for 20-25 paid, naive listeners. This run is unpaid
colleagues, which is a weaker design in two specific ways, and both are
measurable rather than rhetorical:

  * **Not naive.** Someone who has worked on the method can recognise what a
    protected clip is, and may rate it generously or harshly. The page asks
    directly, and `naive` is carried on every row. The default is to report the
    naive and non-naive subgroups SEPARATELY whenever both exist, because
    pooling them would hide exactly the bias that matters.
  * **Small n.** With a handful of listeners the unit of analysis is the
    listener, not the rating, and a confidence interval over 5 listeners is
    wide. The script refuses to print a CI below `--min-listeners` and says why.

Nothing here converts a convenience panel into a probability sample. The
contribution of this test is bounded: it can show that the shipping arm is not
catastrophically unnatural and is still recognisable as the same speaker, which
is what the paper claims. It cannot support a precise MOS.

THE SCREENER
------------
Two anchors fix the ends of the scale: the clean original must be rated >= 4 and
the 0 dB-SNR noise anchor <= 2. A listener who fails either was not doing the
task as instructed -- or had a broken playback chain -- and the page has already
computed `screen_pass` from their own ratings. Failures are EXCLUDED by default
and always counted in the output; `--keep-failed` includes them, for checking
whether a conclusion depends on the exclusion.

Run:
  python experiments/analyse_listening.py
  python experiments/analyse_listening.py --keep-failed --min-listeners 3
Output:
  experiments/results/listening_ratings.csv   (tidy, unblinded, one row/rating)
  experiments/results/listening_summary.csv   (per condition x block)
"""
import argparse
import glob
import os
import sys

import warnings

import numpy as np
import pandas as pd

# scipy warns "Sample size too small for normal approximation" on every paired
# test at this n. The small n is the headline of this analysis, stated in the
# output and in the suppressed CIs; the per-call warning adds nothing and buries
# the tables.
warnings.filterwarnings("ignore", message=".*too small for normal approximation.*")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

LT = os.path.join(PROJ, "listening_test")
R = os.path.join(PROJ, "experiments", "results")

REQUIRED = ["listener", "block", "stim_id", "rating"]
# Printed in this order; the shipping arm last so it is the line the eye lands on.
COND_ORDER = ["original", "plain", "nsT", "full", "fullnsT", "anchor_noise"]


def percentile_ci(x, alpha=0.05, n_boot=2000, seed=20261005):
    """PERCENTILE bootstrap CI for a mean, with the listener as the unit.

    RENAMED 2026-10-07 (council v5 code review). This was called `bca_ci`, but
    it is not BCa: there is no bias correction (z0) and no jackknife
    acceleration, which are the two things that make an interval BCa. Five
    other scripts import a genuine `bca_ci` from analyse_matched_control under
    that exact name, and RESULTS says "BCa 95 % CI" in three places, so the
    collision would have mislabelled every listening interval the moment the
    panel's data arrived. The docstring was always honest; the name was not.

    A percentile interval is the right choice here and is kept: at the panel
    sizes this script is built for (often < 10 listeners) the jackknife
    acceleration is estimated from too few points to help, and BCa can be
    worse than percentile in that regime. The point is to say which one it is.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return float(np.percentile(b, 100 * alpha / 2)), float(
        np.percentile(b, 100 * (1 - alpha / 2)))


def wilcoxon_p(d):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 1 or np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, zero_method="wilcox",
                              alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=os.path.join(LT, "responses"))
    ap.add_argument("--key", default=os.path.join(LT, "stimuli_key.csv"))
    ap.add_argument("--keep-failed", action="store_true", dest="keep_failed",
                    help="include listeners who failed the anchor screener")
    ap.add_argument("--min-listeners", type=int, default=4,
                    dest="min_listeners",
                    help="below this, per-condition CIs are suppressed as "
                         "uninformative rather than printed")
    ap.add_argument("--pool-naive", action="store_true", dest="pool_naive",
                    help="pool naive and non-naive listeners into one table. "
                         "Off by default: pooling hides the bias this panel "
                         "actually has.")
    ap.add_argument("--out-ratings", default=os.path.join(R, "listening_ratings.csv"),
                    dest="out_ratings")
    ap.add_argument("--out-summary", default=os.path.join(R, "listening_summary.csv"),
                    dest="out_summary")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.responses, "*.csv")))
    if not files:
        print(f"[L] no response files in {a.responses}")
        print("[L] Nothing to analyse yet. The stimuli, the blinded page and "
              "the screener are built; this script is what runs once listeners "
              "have returned their CSVs. Drop each returned file into that "
              "directory and re-run.")
        return

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as exc:
            print(f"[L] SKIP {os.path.basename(f)}: unreadable ({exc})")
            continue
        miss = [c for c in REQUIRED if c not in d.columns]
        if miss:
            print(f"[L] SKIP {os.path.basename(f)}: missing columns {miss}")
            continue
        d["source_file"] = os.path.basename(f)
        frames.append(d)
    if not frames:
        raise SystemExit("[L] no usable response files")
    raw = pd.concat(frames, ignore_index=True)

    # One listener, one submission. A duplicate id across files is ambiguous and
    # is reported rather than silently averaged.
    dup = (raw.groupby("listener")["source_file"].nunique() > 1)
    if dup.any():
        print(f"[L] WARNING: listener id(s) {list(dup[dup].index)} appear in "
              f"more than one file; each submission is kept separately, keyed "
              f"by (listener, source_file).")
        raw["listener"] = raw.listener.astype(str) + "@" + raw.source_file
    raw["listener"] = raw.listener.astype(str)

    key = pd.read_csv(a.key)
    key["stim_id"] = key.stim_id.astype(str)
    raw["stim_id"] = raw.stim_id.astype(str)
    unknown = sorted(set(raw.stim_id) - set(key.stim_id))
    if unknown:
        raise SystemExit(
            f"[L] {len(unknown)} stim_id(s) in the responses are not in "
            f"{os.path.basename(a.key)}, first {unknown[0]}. The responses were "
            f"collected against a DIFFERENT stimulus build than this key -- "
            f"check build_seed. Refusing to unblind against the wrong key.")
    if "build_seed" in raw.columns:
        seeds = sorted(set(raw.build_seed.dropna().astype(str)))
        if len(seeds) > 1:
            raise SystemExit(f"[L] responses span {len(seeds)} stimulus builds "
                             f"({seeds}); they are not comparable.")

    df = raw.merge(key[["stim_id", "speaker", "condition", "arm"]],
                   on="stim_id", how="left")

    # ----------------------------------------------------------- the screener
    if "screen_pass" in df.columns:
        per = df.groupby("listener")["screen_pass"].max()
    else:
        per = pd.Series(dtype=float)
    failed = sorted(per[per != 1].index) if len(per) else []
    n_all = df.listener.nunique()
    print("=" * 78)
    print("LISTENING TEST")
    print("=" * 78)
    print(f"{len(files)} file(s), {n_all} listener submission(s), "
          f"{len(df)} ratings")
    if len(per):
        print(f"screener: {int((per == 1).sum())} passed, {len(failed)} failed "
              f"(clean original >= 4 and noise anchor <= 2)")
        if failed:
            print(f"  failed: {', '.join(map(str, failed))}")
    if failed and not a.keep_failed:
        df = df[~df.listener.isin(failed)]
        print(f"  excluded; {df.listener.nunique()} listener(s) remain "
              f"(--keep-failed to include them)")
    elif failed:
        print("  KEPT (--keep-failed): the tables below include listeners who "
              "did not use the scale as instructed")

    if not df.listener.nunique():
        raise SystemExit("[L] no listeners left after screening")

    for col, label in (("headphones", "headphones"), ("naive", "naive")):
        if col in df.columns:
            comp = (df.drop_duplicates("listener")[col]
                    .map({1: "yes", 0: "no", "1": "yes", "0": "no"})
                    .value_counts(dropna=False).to_dict())
            print(f"panel {label}: {comp}")

    df.to_csv(a.out_ratings, index=False)

    # -------------------------------------------------------------- the tables
    groups = [("all listeners", df)]
    if not a.pool_naive and "naive" in df.columns:
        nv = df.naive.astype(str)
        if nv.isin(["1"]).any() and nv.isin(["0"]).any():
            groups = [("naive listeners", df[nv == "1"]),
                      ("non-naive listeners", df[nv == "0"]),
                      ("all listeners (pooled -- see the caveat)", df)]

    BLOCKS = {"A": "naturalness (1 bad - 5 excellent)",
              "B": "same speaker as the reference (1 certainly different - "
                   "5 certainly same)"}
    summary_rows = []
    for gname, g in groups:
        for blk, blabel in BLOCKS.items():
            sub = g[g.block == blk]
            if not len(sub):
                continue
            # Listener-level means first: the listener is the unit, not the
            # rating. With a handful of people, treating 48 ratings from one
            # person as 48 observations would understate the uncertainty by
            # roughly sqrt(48).
            per_lc = (sub.groupby(["listener", "condition"]).rating
                      .mean().reset_index())
            n_lis = per_lc.listener.nunique()
            print("\n" + "-" * 78)
            print(f"{gname}  |  part {blk}: {blabel}")
            print(f"n = {n_lis} listener(s), unit = listener")
            print("-" * 78)
            show_ci = n_lis >= a.min_listeners
            if not show_ci:
                print(f"  CIs suppressed: {n_lis} listener(s) is below "
                      f"--min-listeners {a.min_listeners}. Means only.")
            hdr = f"  {'condition':14s} {'mean':>6s} {'SD':>6s} {'n':>4s}"
            if show_ci:
                hdr += f" {'95% CI':>16s}"
            print(hdr)
            order = [c for c in COND_ORDER if c in set(per_lc.condition)]
            order += [c for c in sorted(set(per_lc.condition)) if c not in order]
            for cond in order:
                v = per_lc[per_lc.condition == cond].rating
                line = (f"  {cond:14s} {v.mean():6.2f} "
                        f"{(v.std() if len(v) > 1 else float('nan')):6.2f} "
                        f"{len(v):4d}")
                lo = hi = float("nan")
                if show_ci:
                    lo, hi = percentile_ci(v.to_numpy())
                    line += f" [{lo:6.2f},{hi:6.2f}]"
                print(line)
                summary_rows.append(dict(group=gname, block=blk, condition=cond,
                                         mean=round(float(v.mean()), 3),
                                         sd=(round(float(v.std()), 3)
                                             if len(v) > 1 else np.nan),
                                         n_listeners=len(v),
                                         ci_lo=round(lo, 3), ci_hi=round(hi, 3)))

            # Paired, per listener, against the two reference points the claim
            # needs: the clean original (how much quality is given up) and the
            # unconstrained PGD arm (whether the constraint costs anything
            # audible).
            piv = per_lc.pivot(index="listener", columns="condition",
                               values="rating")
            for base in ("original", "plain"):
                if base not in piv.columns:
                    continue
                print(f"\n  paired against {base}, per listener:")
                for cond in order:
                    if cond == base or cond not in piv.columns:
                        continue
                    d = (piv[cond] - piv[base]).dropna()
                    if not len(d):
                        continue
                    p = wilcoxon_p(d.to_numpy())
                    extra = ""
                    if show_ci:
                        lo, hi = percentile_ci(d.to_numpy())
                        extra = f"  95% CI [{lo:+.2f},{hi:+.2f}]"
                    print(f"      {cond:14s} {d.mean():+6.2f}  "
                          f"n={len(d):2d}  p={p:.3f}{extra}")

    pd.DataFrame(summary_rows).to_csv(a.out_summary, index=False)

    print("\n" + "=" * 78)
    print("HOW THIS MAY AND MAY NOT BE REPORTED")
    print("=" * 78)
    print("This is an unpaid convenience panel, not the 20-25 paid naive")
    print("listeners council #6 asked for. It can support: the shipping arm is")
    print("not catastrophically unnatural, and it is still heard as the same")
    print("speaker. It cannot support a precise MOS, and the number must be")
    print("reported with the panel size, the recruitment route and the naive")
    print("split stated in the same sentence.")
    print("")
    print("If the naive and non-naive subgroups disagree, the non-naive group")
    print("is the biased one and the paper says so -- it does not quietly")
    print("report whichever is more favourable.")
    print(f"\nsaved {a.out_ratings}")
    print(f"saved {a.out_summary}")


if __name__ == "__main__":
    main()
