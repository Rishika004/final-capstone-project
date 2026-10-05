"""
experiments/analyse_listening.py
================================
Analyse returned listening-test CSVs.

  1. Screen listeners on the two anchors: `original` must average >= 4.0 on
     naturalness and `anchor_noise` <= 2.0. Failing either means the listener
     was not attending or had a broken playback chain; they are excluded and
     reported, not silently dropped.
  2. Per condition: mean naturalness (block A) and mean same-speaker rating
     (block B), each with a 95 % confidence interval computed over LISTENER
     means (listeners are the unit of replication, not individual trials --
     trials within a listener are not independent).
  3. Join against the measured protection rate so the final table is the
     quality/identity/protection trade-off in one place.

Run:
  python experiments/analyse_listening.py listening_test/responses/*.csv
"""
import sys, io, os, glob, argparse
if not getattr(sys.stdout, "_nvp_utf8", False):   # guard: re-wrapping a second
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # time closes the first wrapper
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from utils.csvio import read_csv as read_csv_safe

# protection on the two trained ASV systems at t=200, from RESULTS_ICASSP.md s8
PROTECTION = {"original": 0.0, "anchor_noise": np.nan, "tilt": 0.0,
              "mcadams_095": 17.5, "mcadams_085": 37.5, "mcadams_080": 57.5,
              "additive_plain": 70.0, "additive_eot": 90.0}


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*",
                    default=glob.glob(os.path.join(PROJ, "listening_test", "responses", "*.csv")))
    ap.add_argument("--min-natural-original", type=float, default=4.0)
    ap.add_argument("--max-natural-anchor", type=float, default=2.0)
    args = ap.parse_args()

    files = []
    for pat in args.csvs:
        files.extend(glob.glob(pat))
    if not files:
        print("no response CSVs found -- run the test first "
              "(experiments/make_listening_test.py, then open listening_test/index.html)")
        return
    df = pd.concat([read_csv_safe(f) for f in files], ignore_index=True)
    print(f"{len(files)} files, {df.listener.nunique()} listeners, {len(df)} ratings\n")

    # ---- 1. screening
    a = df[df.block == "A"]
    scr = a.pivot_table(index="listener", columns="condition", values="rating", aggfunc="mean")
    keep, drop = [], []
    for L, row in scr.iterrows():
        ok = (row.get("original", np.nan) >= args.min_natural_original and
              row.get("anchor_noise", np.nan) <= args.max_natural_anchor)
        (keep if ok else drop).append(L)
    print("SCREENING (original >= %.1f, anchor_noise <= %.1f)" %
          (args.min_natural_original, args.max_natural_anchor))
    for L, row in scr.iterrows():
        print(f"  {'keep' if L in keep else 'DROP'}  {L:18s} original={row.get('original', np.nan):.2f} "
              f"anchor={row.get('anchor_noise', np.nan):.2f}")
    if not keep:
        print("\nno listener passed screening; nothing to analyse")
        return
    df = df[df.listener.isin(keep)]
    print(f"\n{len(keep)} listeners retained, {len(drop)} excluded\n")

    # ---- 2. per-condition means over listener means
    rows = []
    for cond in PROTECTION:
        r = {}
        for blk, label in (("A", "naturalness"), ("B", "same_speaker")):
            sub = df[(df.block == blk) & (df.condition == cond)]
            if len(sub) == 0:
                r[label], r[label + "_ci"] = np.nan, np.nan
                continue
            per_listener = sub.groupby("listener").rating.mean()
            r[label] = per_listener.mean()
            r[label + "_ci"] = ci95(per_listener)
        rows.append(dict(condition=cond, protection=PROTECTION[cond], **r))
    out = pd.DataFrame(rows)

    print("=" * 78)
    print("CONDITION MEANS (95 % CI over listeners)")
    print("=" * 78)
    for _, r in out.iterrows():
        nat = ("  n/a" if np.isnan(r.naturalness)
               else f"{r.naturalness:.2f} +-{r.naturalness_ci:.2f}")
        sam = ("  n/a" if np.isnan(r.same_speaker)
               else f"{r.same_speaker:.2f} +-{r.same_speaker_ci:.2f}")
        prot = "  n/a" if np.isnan(r.protection) else f"{r.protection:5.1f} %"
        print(f"  {r.condition:16s} naturalness {nat:14s} same-speaker {sam:14s} protection {prot}")

    # ---- 3. the decisive comparison
    print("\n" + "=" * 78)
    print("THE QUESTION THAT DECIDES VIABILITY")
    print("=" * 78)
    pairs = [("additive_eot", "mcadams_080"), ("additive_plain", "mcadams_085")]
    for x, y in pairs:
        rx = out[out.condition == x].iloc[0]
        ry = out[out.condition == y].iloc[0]
        if np.isnan(rx.same_speaker) or np.isnan(ry.same_speaker):
            continue
        print(f"  {x} ({rx.protection:.0f} % protection): same-speaker {rx.same_speaker:.2f}, "
              f"naturalness {rx.naturalness:.2f}")
        print(f"  {y} ({ry.protection:.0f} % protection): same-speaker {ry.same_speaker:.2f}, "
              f"naturalness {ry.naturalness:.2f}")
        d = rx.same_speaker - ry.same_speaker
        print(f"  -> identity retained {'better' if d > 0 else 'worse'} by {x} "
              f"(difference {d:+.2f})\n")
    print("A protection is only usable if listeners still accept it as the same person.")

    out_csv = os.path.join(PROJ, "experiments", "results", "listening_summary.csv")
    out.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
