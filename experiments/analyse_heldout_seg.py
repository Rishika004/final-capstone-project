"""
experiments/analyse_heldout_seg.py
==================================
Control A at the SHIPPING budget — the arm RESULTS 47e made mandatory.

WHAT IS BEING DECIDED
---------------------
RESULTS 47 (Control A) held the d-vector out of the crafting ensemble and the
reversal disappeared and flipped sign: +0.074/+0.079 matched against
-0.008/-0.012 held out, non-overlapping CIs. OUTCOME A-2.

`PREREG_council5_controls.md` A.2 declared the limitation in advance: those are
the POOLED-partials arms, not the `--seg-partials` shipping recipe, because
`full_m30_seg_nodvec` did not exist. And it declared the consequence:

    "If the control contradicts the matched result, the seg version becomes
     mandatory; if it agrees, the non-seg evidence is sufficient."

It contradicted. `run_heldout_seg.sh` crafts the missing arm. This script reads
it.

THE PREDICTION, COMMITTED BEFORE THE ARM WAS CRAFTED (RESULTS 49e)
------------------------------------------------------------------
RESULTS 49's contrast A measured what switching pooled -> segment does to these
same cells in the MATCHED condition: mean |change| 0.0072, max 0.0192, zero
sign flips. So:

    PREDICTION: the seg held-out grid reproduces the pooled held-out grid to
    within about 0.01 per cell, and NO cell flips sign.

    If it holds       -> 47's OUTCOME A-2 is budget- and crafting-mode-robust.
                         C2's reversal is a matched-encoder observation, full
                         stop, and the seg arm closes 47e.
    If a sign flips   -> something other than budget or crafting mode is
                         operating. 47's reading needs re-examination, and this
                         script says so rather than picking whichever grid
                         agrees with 47.

ENDPOINT, identical to 27 / 45 / 47 / 49:
    mean paired per-speaker (fullnsT - full) in (cos_after_purify - threshold),
    n = 29, BCa 95 % CI (2000 resamples), Wilcoxon, Holm over this control's
    12 cells (3 encoders x 2 purifiers x 2 timesteps).
    NEGATIVE = the constraint helps.  POSITIVE = the reversal.

Run:
    python experiments/analyse_heldout_seg.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.analyse_matched_control import bca_ci, holm, wilcoxon_p

R = os.path.join(PROJ, "experiments", "results")
ENC_ORDER = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]
FAMILY = 12
BLOCKS = [("DiffWave-LibriSpeech", 1, "libri_t1"),
          ("DiffWave-LibriSpeech", 3, "libri_t3"),
          ("DiffWave-SC09", 1, "sc09_t1"),
          ("DiffWave-SC09", 3, "sc09_t3")]

# grid name -> (prefix, full tag, fullnsT tag, label)
GRIDS = {
    "heldout_seg": ("heldout_seg_c2", "full_m30_seg_nodvec",
                    "fullnsT_m30_seg_nodvec",
                    "HELD-OUT, segment partials  (the mandatory arm)"),
    "heldout_pooled": ("heldout_c2", "full_m30_f15_nodvec",
                       "fullnsT_m30_f15_nodvec",
                       "HELD-OUT, pooled partials  (RESULTS 47)"),
    "matched_seg": ("protect_foreign_seg", "full_tc_m30_seg",
                    "fullnsT_tc_m30_seg",
                    "MATCHED, segment partials  (RESULTS 27, the shipping arm)"),
}
# The pre-registered tolerance, from 49's measured crafting-mode contrast.
TOL = 0.01


def grid(prefix, full_tag, nst_tag):
    cells, missing = [], []
    for pur, t, stem in BLOCKS:
        p = os.path.join(R, f"{prefix}_{stem}.csv")
        if not os.path.isfile(p):
            missing.append(f"{prefix}_{stem}.csv")
            continue
        d = pd.read_csv(p)
        if "margin" not in d.columns:
            d["margin"] = d.cos_after_purify - d.threshold
        d["speaker"] = d.file.astype(str).str.extract(r"^spk(\d+)_")[0].astype(int)
        for enc in ENC_ORDER:
            sub = d[(d.encoder == enc) & (d.variant.isin([full_tag, nst_tag]))]
            piv = sub.pivot_table(index="speaker", columns="variant",
                                  values="margin")
            if full_tag not in piv.columns or nst_tag not in piv.columns:
                missing.append(f"{prefix}_{stem}.csv:{enc}")
                continue
            s = piv[[nst_tag, full_tag]].dropna()
            dd = (s[nst_tag] - s[full_tag]).to_numpy()
            lo, hi, _, _ = bca_ci(dd, alpha=0.05, n_boot=2000)
            cells.append(dict(purifier=pur, t=t, encoder=enc, n=len(dd),
                              delta=float(dd.mean()), lo=lo, hi=hi,
                              p=wilcoxon_p(dd), helped=int((dd < 0).sum())))
    return cells, missing


def adjust(cells):
    praw = [c["p"] for c in cells]
    if len(praw) < FAMILY:
        praw = praw + [1.0] * (FAMILY - len(praw))
    for c, pa in zip(cells, holm(praw)[:len(cells)]):
        c["p_holm"] = pa
        c["sig"] = bool(pa < 0.05 and (c["lo"] > 0 or c["hi"] < 0))
    return cells


def show(label, cells):
    print("=" * 106)
    print(label + f"     (n = {cells[0]['n']})")
    print("=" * 106)
    print(f"  {'purifier':22s} {'t':>2} {'encoder':22s} {'delta':>9s} "
          f"{'95% CI':>20s} {'k/n':>7s} {'p_holm':>9s} sig")
    for c in cells:
        print(f"  {c['purifier']:22s} {c['t']:2d} {c['encoder']:22s} "
              f"{c['delta']:+9.4f} [{c['lo']:+8.4f},{c['hi']:+8.4f}] "
              f"{c['helped']:3d}/{c['n']:<3d} {c['p_holm']:9.2e} "
              f"{'*' if c['sig'] else ''}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "heldout_seg_cells.csv"))
    a = ap.parse_args()

    tables, miss = {}, {}
    for k, (pre, ft, nt, _lab) in GRIDS.items():
        cells, m = grid(pre, ft, nt)
        if m:
            miss[k] = m
        if cells:
            tables[k] = adjust(cells)

    for k, m in miss.items():
        print(f"[heldout-seg] MISSING {k}: {', '.join(m[:4])}"
              + (" ..." if len(m) > 4 else ""))
    if "heldout_seg" not in tables or len(tables["heldout_seg"]) != FAMILY:
        n = len(tables.get("heldout_seg", []))
        print(f"\n[heldout-seg] NOT READY: {n} of {FAMILY} cells. The verdict "
              f"is defined on all 12; nothing below is the result.")
        if n == 0:
            return

    rows = []
    for k, (_pre, _ft, _nt, lab) in GRIDS.items():
        if k in tables:
            show(lab, tables[k])
            rows += [dict(grid=k, **c) for c in tables[k]]
    pd.DataFrame(rows).to_csv(a.out, index=False)

    seg = {(c["purifier"], c["t"], c["encoder"]): c
           for c in tables.get("heldout_seg", [])}
    pool = {(c["purifier"], c["t"], c["encoder"]): c
            for c in tables.get("heldout_pooled", [])}
    matched = {(c["purifier"], c["t"], c["encoder"]): c
               for c in tables.get("matched_seg", [])}

    # ------------------------------------------- the pre-registered reading
    if seg and pool:
        keys = sorted(set(seg) & set(pool))
        print("=" * 106)
        print("THE PRE-REGISTERED COMPARISON: held-out pooled  ->  held-out "
              "segment")
        print(f"tolerance declared in 49e: |change| <= {TOL:.3f} per cell, "
              f"and NO sign flip")
        print("=" * 106)
        print(f"  {'purifier':22s} {'t':>2} {'encoder':22s} {'pooled':>10s} "
              f"{'segment':>10s} {'change':>9s} {'flip':>6s} {'>tol':>6s}")
        flips, over = [], []
        for k in keys:
            dp, ds = pool[k]["delta"], seg[k]["delta"]
            ch = ds - dp
            fl = np.sign(dp) != np.sign(ds)
            ov = abs(ch) > TOL
            if fl:
                flips.append(k)
            if ov:
                over.append((k, ch))
            print(f"  {k[0]:22s} {k[1]:2d} {k[2]:22s} {dp:+10.4f} "
                  f"{ds:+10.4f} {ch:+9.4f} {'SIGN' if fl else '':>6s} "
                  f"{'yes' if ov else '':>6s}")
        ch_all = np.array([seg[k]["delta"] - pool[k]["delta"] for k in keys])
        print(f"\n  mean |change| {np.abs(ch_all).mean():.4f}, "
              f"max |change| {np.abs(ch_all).max():.4f}, "
              f"{len(flips)} sign flips, {len(over)} cells over tolerance")

        # ------------------------------------------------- the verdict
        print("\n" + "=" * 106)
        print("VERDICT -- 47e")
        print("=" * 106)
        dvec_t1 = [k for k in keys
                   if k[2] == "Resemblyzer d-vector" and k[1] == 1]
        seg_signs = [seg[k]["delta"] for k in dvec_t1]
        print("  d-vector, t = 1, the cells 47's OUTCOME A-2 turns on:")
        for k in dvec_t1:
            c = seg[k]
            m = matched.get(k)
            print(f"    {k[0]:22s}  held-out seg {c['delta']:+.4f} "
                  f"[{c['lo']:+.4f},{c['hi']:+.4f}]"
                  + (f"   matched seg {m['delta']:+.4f}" if m else ""))
        if flips:
            print("\n  SIGN FLIP in: " + "; ".join(f"{k[2]}/{k[0]}/t={k[1]}"
                                                   for k in flips))
            print("  The pooled and segment held-out grids DISAGREE in sign.")
            print("  47's reading is NOT crafting-mode-robust. Report both")
            print("  grids and do NOT pick the one that agrees with 47.")
        elif all(s <= 0 for s in seg_signs):
            print("\n  A-2 CONFIRMED AT THE SHIPPING BUDGET. Held out, the")
            print("  d-vector t=1 cells are non-positive under segment")
            print("  crafting too, so the reversal is a property of crafting")
            print("  against the d-vector and not of the constraint, at both")
            print("  budgets measured. 47e is CLOSED and C2's reversal stays")
            print("  demoted to a matched-encoder observation.")
        else:
            print("\n  The d-vector t=1 cells are POSITIVE under segment")
            print("  crafting while negative under pooled. No sign flip was")
            print("  predicted; this is outside the pre-registered space and")
            print("  must be reported as such, not resolved by choosing a")
            print("  grid.")
        if over:
            print(f"\n  NOTE: {len(over)} cells exceed the 0.01 tolerance "
                  f"(max {max(abs(c) for _k, c in over):.4f}). The tolerance")
            print("  came from the MATCHED crafting-mode contrast (49A); the")
            print("  held-out condition was never claimed to be as quiet.")

    if seg and matched:
        print("\n" + "=" * 106)
        print("THE CONTRAST THAT IS THE RESULT: matched vs held-out, both "
              "segment, one variable")
        print("=" * 106)
        print(f"  {'purifier':22s} {'t':>2} {'encoder':22s} {'matched':>10s} "
              f"{'held-out':>10s} {'swing':>9s}")
        for k in sorted(set(seg) & set(matched)):
            print(f"  {k[0]:22s} {k[1]:2d} {k[2]:22s} "
                  f"{matched[k]['delta']:+10.4f} {seg[k]['delta']:+10.4f} "
                  f"{seg[k]['delta'] - matched[k]['delta']:+9.4f}")

    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
