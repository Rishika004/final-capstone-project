"""
experiments/analyse_n5.py
=========================
N5' -- the budget-matched attribution, through DiffVC, on the SHIPPING recipe.

PREREG: paper/PREREG_N5_budget_matched_plain.md (and its 2026-11-12 amendment)

THE POINT. RESULTS 59 could not attribute the protection to purifier-in-the-
loop crafting, for two different reasons in the two conditions it had:
  * matched m30_f15   -- both arms saturated, 0/9 cells, not identifiable;
  * held-out m30_f15  -- 7/9 cells, but the purifier-aware arm spent 2.89 dB
                         more of the budget (59f), so "protects better" and
                         "perturbs more" were confounded.
The segment-crafted shipping arms have neither problem: 60d measured them at
16.206 vs 16.113 dB SNR (paired -0.093 dB, p = 0.408) with the purifier-aware
arm at BETTER PESQ (+0.192, p = 1.6e-06). They were never run through DiffVC.
This script reads that run.

Primary cell, fixed in the prereg before the data: ECAPA-TDNN, channel `ours`,
fullnsT_tc_m30_seg MINUS plain_tc_m30_seg, paired over 29 speakers.

UNFILTERED margins throughout. The AntiFake-style pre-filter that
exp_cloning_protection prints in its own summary reduces n per cell and the
project's headline convention (W1 2a) is the unfiltered paired margin; the
filtered binary rates are reported separately with their n.

Run:
  python experiments/analyse_n5.py
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

DEFAULT_IN = os.path.join(PROJ, "experiments", "results",
                          "cloning_tc_m30_seg_n5.csv")
SHIP, PLAIN, BASE = "fullnsT_tc_m30_seg", "plain_tc_m30_seg", "unprotected"
ENCS = ["ECAPA-TDNN", "WavLM-SV", "Resemblyzer d-vector"]
PURS = ["none", "ours", "diffwave"]
PRIMARY = ("ECAPA-TDNN", "ours")
FLOOR_PCT = 10.0
NOISE_FLOOR = 0.0247            # RESULTS 34, median over the BPDA-null cells

# Measured in 60d; printed so the matching is visible beside the result.
SNR = {PLAIN: 16.206, SHIP: 16.113}
PESQ = {PLAIN: 2.091, SHIP: 2.283}


def paired(d, a, b, enc, pur):
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
    return dict(encoder=enc, purifier=pur, n=len(j), delta=float(diff.mean()),
                lo=float(lo), hi=float(hi), p=float(p), p_w=float(pw),
                better=int((diff < 0).sum()))


def rate(d, v, enc, pur):
    s = d[(d.encoder == enc) & (d.purifier == pur) & (d.variant == v)]
    return (100.0 * s.clone_succeeds.mean(), len(s)) if len(s) else (np.nan, 0)


def main():
    global SHIP, PLAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--ship", default=SHIP)
    ap.add_argument("--plain", default=PLAIN)
    ap.add_argument("--budget-csv", default="", dest="budget_csv",
                    help="spend table from analyse_n1_budget.py --out; when "
                         "given, the matching check is computed from it "
                         "instead of the 60d constants")
    args = ap.parse_args()
    SHIP, PLAIN = args.ship, args.plain
    d = pd.read_csv(args.inp)

    # The matching check must describe THESE arms. Fall back to 60d's measured
    # constants only when the arms are the ones 60d measured.
    if args.budget_csv and os.path.isfile(args.budget_csv):
        q = pd.read_csv(args.budget_csv)
        for arm in (PLAIN, SHIP):
            s = q[q.variant == arm]
            if len(s):
                SNR[arm] = float(s.snr_db.mean())
                PESQ[arm] = float(s.pesq.mean())
    missing = [a for a in (PLAIN, SHIP) if a not in SNR]
    if missing:
        raise SystemExit(
            f"no realised-budget numbers for {missing}. Run\n"
            f"  python experiments/analyse_n1_budget.py --arms "
            f"{PLAIN},{SHIP} --pair {PLAIN}:{SHIP}:PAIR --out <csv>\n"
            f"and pass it with --budget-csv. Refusing to print a matching "
            f"check for arms it was not measured on.")

    print("N5' -- budget-matched attribution on the SHIPPING recipe, DiffVC")
    print(f"   file {os.path.basename(args.inp)}   "
          f"speakers {d.speaker.nunique()}   arms {sorted(d.variant.unique())}")
    print("   margin = clone similarity - threshold; NEGATIVE = clone fails\n")

    print("=" * 74)
    print("THE MATCHING CHECK (prereg 4; tolerance 0.5 dB) -- from 60d")
    print("=" * 74)
    print(f"  {PLAIN:<22} SNR {SNR[PLAIN]:.3f} dB   PESQ {PESQ[PLAIN]:.3f}")
    print(f"  {SHIP:<22} SNR {SNR[SHIP]:.3f} dB   PESQ {PESQ[SHIP]:.3f}")
    gap = abs(SNR[SHIP] - SNR[PLAIN])
    matched = gap < 0.5
    print(f"  gap {gap:.3f} dB -> "
          f"{'within tolerance' if matched else 'MATCH FAILED (prereg 5-D)'}")
    print("  The paired significance test lives in analyse_n1_budget.py (1);")
    print("  this line checks the 0.5 dB tolerance only. Read both.")
    if PESQ.get(SHIP, 0) >= PESQ.get(PLAIN, 0):
        print("  The purifier-aware arm has the BETTER PESQ, so it is not")
        print("  buying its advantage with perceptual cost.")
    else:
        print(f"  NOTE: the purifier-aware arm has WORSE PESQ "
              f"({PESQ[SHIP]:.3f} vs {PESQ[PLAIN]:.3f}) -- equal SNR does not "
              f"mean equal perceptual cost; say so beside any result.")

    print("\n" + "=" * 74)
    print("(a) sanity: does the defence work at all here? arm MINUS unprotected")
    print("=" * 74)
    res_a = [r for pur in PURS for e in ENCS
             if (r := paired(d, BASE, SHIP, e, pur)) is not None]
    for r, pa in zip(res_a, holm([x["p"] for x in res_a])):
        ru, _ = rate(d, BASE, r["encoder"], r["purifier"])
        rp, n = rate(d, SHIP, r["encoder"], r["purifier"])
        star = "***" if pa < .001 else "**" if pa < .01 else "*" if pa < .05 else ""
        print(f"  {r['purifier']:<9}{r['encoder']:<22}{r['delta']:>+8.4f}"
              f"  [{r['lo']:+.4f},{r['hi']:+.4f}]  {ru:>5.1f}% -> {rp:>5.1f}%"
              f"  {r['better']:>2}/{r['n']}  p_holm={pa:.3g} {star}")

    print("\n" + "=" * 74)
    print(f"(b) PRIMARY: {SHIP} MINUS {PLAIN}, at MATCHED budget")
    print("    negative = purifier-in-the-loop earns its place")
    print("=" * 74)
    res = [r for pur in PURS for e in ENCS
           if (r := paired(d, PLAIN, SHIP, e, pur)) is not None]
    padj = holm([x["p"] for x in res])
    n_sig = n_floor = 0
    for r, pa in zip(res, padj):
        r["p_holm"] = float(pa)
        rpl, _ = rate(d, PLAIN, r["encoder"], r["purifier"])
        rsh, n = rate(d, SHIP, r["encoder"], r["purifier"])
        floored = (rpl < FLOOR_PCT) and (rsh < FLOOR_PCT)
        n_floor += floored
        n_sig += (pa < .05) and not floored
        star = "***" if pa < .001 else "**" if pa < .01 else "*" if pa < .05 else ""
        tag = "  <- AT THE FLOOR, not identifiable" if floored else ""
        mark = "  <== PRIMARY" if (r["encoder"], r["purifier"]) == PRIMARY else ""
        print(f"  {r['purifier']:<9}{r['encoder']:<22}{r['delta']:>+8.4f}"
              f"  [{r['lo']:+.4f},{r['hi']:+.4f}]  {rpl:>5.1f}% -> {rsh:>5.1f}%"
              f"  {r['better']:>2}/{r['n']}  p_holm={pa:.3g} {star}{tag}{mark}")
    mx = max(abs(r["delta"]) for r in res)
    print(f"\n  {n_sig} of {len(res)} cells significant and above the floor; "
          f"{n_floor} not identifiable.")
    print(f"  largest |delta| {mx:.4f} "
          f"({'above' if mx > NOISE_FLOOR else 'BELOW'} the {NOISE_FLOOR} "
          f"noise floor, RESULTS 34)")

    print("\n" + "=" * 74)
    print("(c) THE BRANCH (prereg 5), read on the primary cell")
    print("=" * 74)
    pr = next(r for r in res
              if (r["encoder"], r["purifier"]) == PRIMARY)
    rpl, _ = rate(d, PLAIN, PRIMARY[0], PRIMARY[1])
    rsh, _ = rate(d, SHIP, PRIMARY[0], PRIMARY[1])
    print(f"  {PRIMARY[0]} at `{PRIMARY[1]}`: delta {pr['delta']:+.4f} "
          f"[{pr['lo']:+.4f},{pr['hi']:+.4f}]  p_holm={pr['p_holm']:.3g}")
    print(f"  clone success {rpl:.1f} % -> {rsh:.1f} %")
    if (rpl < FLOOR_PCT) and (rsh < FLOOR_PCT):
        print("\n  The primary cell is AT THE FLOOR: both arms clone at under")
        print(f"  {FLOOR_PCT:.0f} %, so the binary comparison carries no")
        print("  information and the branch must be read from the continuous")
        print("  margin alone, stated as such. This is 59b's situation and the")
        print("  prereg's 5-D spirit: report it, do not dress it as a null.")
    # THE GATE IS A GATE. The prereg says "no verdict is read from a failed
    # match", and the first version of this script printed 5-A anyway on a run
    # whose arms were 2.878 dB apart. A branch label on a confounded contrast
    # is worse than no label, because it travels.
    if not matched:
        print("\n  => 5-D / 7-D: THE BUDGET MATCH FAILED. No verdict.")
        print(f"     The arms are {gap:.3f} dB apart, so 'protects better' and")
        print("     'perturbs more' are not separated by this run and the")
        print("     cells above must NOT be read as an attribution. They are")
        print("     reported as a confounded contrast or not at all.")
        return
    if pr["p_holm"] < .05 and pr["delta"] < 0:
        v = "5-A: the effect SURVIVES at matched budget"
    elif pr["delta"] < 0:
        v = "5-B: direction holds, significance does not"
    else:
        v = "5-C: the effect does NOT survive -- 59c was a budget effect"
    print(f"\n  => {v}")
    print("\n  Secondary cells decide nothing on their own; the prereg named")
    print("  one primary cell precisely so the branch could not be shopped.")


if __name__ == "__main__":
    main()
