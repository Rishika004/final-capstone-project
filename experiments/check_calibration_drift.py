"""
experiments/check_calibration_drift.py
======================================
Does a recalibration reproduce the frozen operating point?

Every `_v2` calibration written by `run_asv_conventions.sh` has a frozen twin
that the paper's numbers were actually scored against. If the two disagree, one
of three things is true and all of them matter:

  * the channel is not deterministic (a seed escaped),
  * the code path changed since the frozen file was written, or
  * the frozen file was produced with different flags than the script claims.

None of those is a rounding question, so this check **fails** rather than warns.
It is also the cheapest possible test of whether the paper's operating point is
reproducible from HEAD — the clean channel already passed it to four decimals
(RESULTS §42a), and this extends that to every channel the paper reports.

Tolerance: thresholds and EERs are written rounded to 4 decimals by
`calibrate_thresholds.py`, so exact equality at that precision is the bar. A
difference of one unit in the last place is still a difference and is shown.

Run:
  python experiments/check_calibration_drift.py
  python experiments/check_calibration_drift.py --tol 1e-4
"""
import argparse
import glob
import os
import sys

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(PROJ, "experiments", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.0,
                    help="allowed absolute difference in threshold and EER; "
                         "0 means the rounded values must match exactly")
    a = ap.parse_args()

    pairs = []
    for new in sorted(glob.glob(os.path.join(R, "thresholds_*_v2.csv"))):
        old = new[: -len("_v2.csv")] + ".csv"
        if os.path.isfile(old):
            pairs.append((old, new))
    if not pairs:
        print("[drift] no *_v2 calibrations to compare -- nothing to check")
        return 0

    bad, rows = [], []
    for old, new in pairs:
        o = pd.read_csv(old).set_index("encoder")
        n = pd.read_csv(new).set_index("encoder")
        name = os.path.basename(old).replace("thresholds_", "").replace(".csv", "")
        for enc in o.index:
            if enc not in n.index:
                bad.append(f"{name}: {enc} missing from the recalibration")
                continue
            dt = float(n.loc[enc, "threshold"]) - float(o.loc[enc, "threshold"])
            de = float(n.loc[enc, "eer"]) - float(o.loc[enc, "eer"])
            rows.append((name, enc, float(o.loc[enc, "threshold"]),
                         float(n.loc[enc, "threshold"]), dt,
                         float(o.loc[enc, "eer"]), de))
            if abs(dt) > a.tol or abs(de) > a.tol:
                bad.append(f"{name}: {enc} threshold {o.loc[enc,'threshold']} "
                           f"-> {n.loc[enc,'threshold']} ({dt:+.4f}), "
                           f"EER {o.loc[enc,'eer']} -> {n.loc[enc,'eer']} "
                           f"({de:+.4f})")

    w = max(len(r[0]) for r in rows)
    print("=" * (w + 62))
    print(f"CALIBRATION DRIFT: recomputed vs frozen, {len(pairs)} channel(s)")
    print("=" * (w + 62))
    print(f"  {'channel':{w}s} {'encoder':22s} {'frozen':>8s} {'new':>8s} "
          f"{'d(thr)':>8s} {'d(EER)':>8s}")
    for name, enc, to, tn, dt, eo, de in rows:
        flag = "" if (abs(dt) <= a.tol and abs(de) <= a.tol) else "  <-- DRIFT"
        print(f"  {name:{w}s} {enc[:22]:22s} {to:8.4f} {tn:8.4f} "
              f"{dt:+8.4f} {de:+8.4f}{flag}")

    if bad:
        print(f"\n[drift] FAIL: {len(bad)} mismatch(es)")
        for b in bad[:10]:
            print(f"  {b}")
        print("\n  A recalibration that does not reproduce the frozen operating")
        print("  point means the paper's numbers cannot be regenerated from")
        print("  HEAD. Investigate before using either file.")
        return 1

    print(f"\n[drift] PASS: all {len(rows)} (channel, encoder) operating points "
          f"reproduce exactly.")
    print("  The paper's operating points are regenerable from HEAD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
