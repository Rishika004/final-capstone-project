"""
experiments/analyse_n1_plain_vs_eot.py
======================================
N1 -- does purifier-in-the-loop crafting beat PLAIN crafting at the same
perceptual budget, on the CURRENT base, end-to-end through DiffVC?

This is the central table of a defence paper and it had never been run. Both
input CSVs have been on disk since the loud-budget runs; neither is cited
anywhere in RESULTS_ICASSP.md. (The same defect class the 2026-10-04 council
found: rebased CSVs present and unanalysed.)

    cloning_tc_m30_deploy.csv   MATCHED   -- every encoder in the ensemble
    cloning_tc_m30_nodvec.csv   HELD-OUT  -- d-vector NOT in the ensemble

Both are 29 cross-session test-clean speakers, three channels (none / ours =
mel DiffPure v7 / diffwave at t=3), budget m30_f15, DiffVC end-to-end, scored
against a different-chapter clean enrolment at the cross-session EER
thresholds.

WHAT IS AND IS NOT COMPARABLE HERE. Both files carry `plain_m30_f15_*` and
`fullnsT_m30_f15_*` at the SAME budget, so the contrast changes one variable:
whether the purifier was in the crafting loop (plus the noise-matched
regulariser, which the held-out file separates via `full_m30_f15_nodvec`).
Neither file is segment-crafted, so this is NOT the shipping recipe's own
contrast; RESULTS 49 measured crafting mode at 0.0072, so the shipping arm is
unlikely to differ much, but it is not measured and must not be asserted.

THE FLOOR. In the matched condition ECAPA sits at 0-7 % clone success for both
arms, so a null there is saturation, not evidence. This script therefore prints
the success rates beside every contrast and flags cells where both arms are
below `--floor` (default 10 %), so no cell is read as "no difference" when it
is really "no room".

Run:
  python experiments/analyse_n1_plain_vs_eot.py
"""
import sys, io, os, argparse

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from experiments.analyse_matched_control import bca_ci, holm

ENCS = ["ECAPA-TDNN", "WavLM-SV", "Resemblyzer d-vector"]
PURS = ["none", "ours", "diffwave"]

# RESULTS 34: the empirical noise floor for a crafting-loop edit, taken from the
# cell the gate names rather than from the smallest of 18. RESULTS 50: the loop
# does not reproduce itself at fixed seed, worst 0.0328.
NOISE_FLOOR_MEDIAN = 0.0247
NONDETERMINISM_WORST = 0.0328

CASES = [
    dict(file="cloning_tc_m30_deploy.csv", label="MATCHED (every encoder crafted against)",
         plain="plain_m30_f15_all", eot="fullnsT_m30_f15_all", mid=None),
    dict(file="cloning_tc_m30_nodvec.csv", label="HELD-OUT (d-vector not in the ensemble)",
         plain="plain_m30_f15_nodvec", eot="fullnsT_m30_f15_nodvec",
         mid="full_m30_f15_nodvec"),
]


def paired(d, a, b, enc, pur):
    """b MINUS a on the per-speaker margin. Negative = b protects better."""
    s = d[(d.encoder == enc) & (d.purifier == pur)]
    x = s[s.variant == a].set_index("speaker").margin
    y = s[s.variant == b].set_index("speaker").margin
    j = x.to_frame("a").join(y.to_frame("b"), how="inner").dropna()
    if len(j) < 6:
        return None
    diff = (j.b - j.a).to_numpy()
    t, p = ttest_rel(j.b, j.a)
    try:
        _, pw = wilcoxon(j.b, j.a)
    except ValueError:
        pw = float("nan")
    lo, hi = bca_ci(diff)[:2]
    return dict(delta=float(diff.mean()), lo=float(lo), hi=float(hi),
                p=float(p), p_w=float(pw), n=len(j),
                better=int((diff < 0).sum()))


def rate(d, variant, enc, pur):
    s = d[(d.encoder == enc) & (d.purifier == pur) & (d.variant == variant)]
    if not len(s):
        return float("nan"), 0
    return 100.0 * s.clone_succeeds.mean(), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=10.0,
                    help="both arms below this clone-success %% => not identifiable")
    args = ap.parse_args()

    print("N1 -- purifier-in-the-loop vs PLAIN crafting, same budget (m30_f15),")
    print("     current base, 29 cross-session test-clean speakers, DiffVC.")
    print("     margin = clone similarity - threshold; NEGATIVE = clone fails.")
    print(f"     noise floor for a crafting-loop edit: median {NOISE_FLOOR_MEDIAN} "
          f"(RESULTS 34), worst non-determinism {NONDETERMINISM_WORST} (RESULTS 50).\n")

    for case in CASES:
        path = os.path.join(PROJ, "experiments", "results", case["file"])
        if not os.path.isfile(path):
            print(f"!! missing {path}")
            continue
        d = pd.read_csv(path)
        print("=" * 78)
        print(f"{case['label']}")
        print(f"  {case['file']}  |  {d.speaker.nunique()} speakers  |  "
              f"arms: {', '.join(sorted(d.variant.unique()))}")
        print("=" * 78)

        # --- protection against UNPROTECTED: does the defence work at all? ---
        print("\n  (a) the defence itself: shipping arm vs UNPROTECTED")
        res_a = []
        for pur in PURS:
            for e in ENCS:
                r = paired(d, "unprotected", case["eot"], e, pur)
                if r:
                    r.update(pur=pur, enc=e)
                    res_a.append(r)
        if res_a:
            for r, pa in zip(res_a, holm([x["p"] for x in res_a])):
                ru, _ = rate(d, "unprotected", r["enc"], r["pur"])
                rp, n = rate(d, case["eot"], r["enc"], r["pur"])
                star = ("***" if pa < .001 else "**" if pa < .01
                        else "*" if pa < .05 else "")
                print(f"    {r['pur']:<9}{r['enc']:<22}{r['delta']:>+8.4f}"
                      f"  [{r['lo']:+.4f},{r['hi']:+.4f}]"
                      f"  success {ru:>5.1f}% -> {rp:>5.1f}%"
                      f"  {r['better']:>2}/{r['n']}  p_holm={pa:.4g} {star}")

        # --- the attribution: is it the purifier in the loop? ---
        print(f"\n  (b) THE ATTRIBUTION: {case['eot']} MINUS {case['plain']}")
        print("      (negative = purifier-in-the-loop earns its place)")
        res_b = []
        for pur in PURS:
            for e in ENCS:
                r = paired(d, case["plain"], case["eot"], e, pur)
                if r:
                    r.update(pur=pur, enc=e)
                    res_b.append(r)
        if res_b:
            padj = holm([x["p"] for x in res_b])
            n_sig = n_floor = 0
            for r, pa in zip(res_b, padj):
                rpl, _ = rate(d, case["plain"], r["enc"], r["pur"])
                reo, n = rate(d, case["eot"], r["enc"], r["pur"])
                floored = (rpl < args.floor) and (reo < args.floor)
                n_floor += floored
                sig = pa < .05
                n_sig += sig and not floored
                star = ("***" if pa < .001 else "**" if pa < .01
                        else "*" if pa < .05 else "")
                tag = "  <- AT THE FLOOR, not identifiable" if floored else ""
                print(f"    {r['pur']:<9}{r['enc']:<22}{r['delta']:>+8.4f}"
                      f"  [{r['lo']:+.4f},{r['hi']:+.4f}]"
                      f"  success {rpl:>5.1f}% -> {reo:>5.1f}%"
                      f"  {r['better']:>2}/{r['n']}  p_holm={pa:.4g} {star}{tag}")
            mx = max(abs(r["delta"]) for r in res_b)
            print(f"\n    {n_sig} of {len(res_b)} cells significant and above the "
                  f"floor; {n_floor} not identifiable (both arms < {args.floor:.0f}%).")
            print(f"    largest |delta| = {mx:.4f} "
                  f"({'ABOVE' if mx > NOISE_FLOOR_MEDIAN else 'BELOW'} the "
                  f"{NOISE_FLOOR_MEDIAN} noise floor)")

        # --- separate EOT from the regulariser where the file allows ---
        if case["mid"]:
            print(f"\n  (c) splitting it: {case['mid']} (EOT, no reg) MINUS "
                  f"{case['plain']}")
            res_c = []
            for pur in PURS:
                for e in ENCS:
                    r = paired(d, case["plain"], case["mid"], e, pur)
                    if r:
                        r.update(pur=pur, enc=e)
                        res_c.append(r)
            for r, pa in zip(res_c, holm([x["p"] for x in res_c])):
                star = "*" if pa < .05 else ""
                print(f"    {r['pur']:<9}{r['enc']:<22}{r['delta']:>+8.4f}"
                      f"  [{r['lo']:+.4f},{r['hi']:+.4f}]"
                      f"  {r['better']:>2}/{r['n']}  p_holm={pa:.4g} {star}")
            print(f"\n      and the regulariser alone: {case['eot']} MINUS "
                  f"{case['mid']}")
            res_d = []
            for pur in PURS:
                for e in ENCS:
                    r = paired(d, case["mid"], case["eot"], e, pur)
                    if r:
                        r.update(pur=pur, enc=e)
                        res_d.append(r)
            for r, pa in zip(res_d, holm([x["p"] for x in res_d])):
                star = "*" if pa < .05 else ""
                print(f"    {r['pur']:<9}{r['enc']:<22}{r['delta']:>+8.4f}"
                      f"  [{r['lo']:+.4f},{r['hi']:+.4f}]"
                      f"  {r['better']:>2}/{r['n']}  p_holm={pa:.4g} {star}")
        print("")


if __name__ == "__main__":
    main()
