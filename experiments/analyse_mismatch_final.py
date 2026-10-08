"""
experiments/analyse_mismatch_final.py
=====================================
The budget x crafting attribution, which `analyse_budget_2x2.py` could not do.

WHY THIS FILE EXISTS
--------------------
`run_rebase_mismatch_m30_seg.sh` has ended with

    say "DONE. Read the 2x2 with: python experiments/analyse_mismatch_final.py"

since 2026-09-22, and the file has never existed.  Council v4 item E5 asks for
it.  `analyse_budget_2x2.py` says so itself, in its own docstring:

    CAVEAT ... this is NOT a 2x2.  The m20 arms were crafted WITH
    --seg-partials and the m30 arms WITHOUT, so the two rows are the DIAGONAL
    of a budget x crafting table.  What a difference between them proves is
    that {budget, crafting} together matter -- not which one.

The off-diagonal cell (m30 + seg) was run and is on disk.  Three of the four
corners therefore exist, and three corners are enough to attribute the effect,
because each comparison moves exactly one factor:

    margin 30 / floor 15, pooled   (full_tc_m30,     fullnsT_tc_m30)
    margin 30 / floor 15, segment  (full_tc_m30_seg, fullnsT_tc_m30_seg)  <- ships
    margin 20 / floor 20, segment  (full_tc_m20,     fullnsT_tc_m20)

    A. m30 pooled -> m30 seg   CRAFTING MODE at fixed budget
    B. m20 seg    -> m30 seg   BUDGET at fixed crafting mode

The fourth corner (m20 + pooled) is **not** on disk and is **not** scheduled:
with A and B both measured, it would only buy an interaction term that no claim
in the paper rests on.  That is stated here rather than left as an absence.

ENDPOINT, identical to RESULTS 27 / 45 / 47 so the numbers are comparable:
    mean paired per-speaker difference in (cos_after_purify - threshold),
    fullnsT - full, per (purifier, t, encoder) cell.
    unit      speaker, n = 29
    sign      NEGATIVE = the constraint improves protection
              POSITIVE = the reversal
    interval  BCa bootstrap 95 % CI, 2000 resamples
    p         Wilcoxon signed-rank, two-sided
    family    Holm over the 12 cells WITHIN each recipe, declared below

A NOTE ON THE MULTIPLICITY FAMILY, which is the thing councils keep asking for
-----------------------------------------------------------------------------
Each recipe is corrected within its own 12 cells (3 encoders x 2 purifiers x
2 timesteps).  The three recipes are NOT pooled into one family of 36.  The
reason is that the shipping recipe's 12 cells are the pre-specified confirmatory
family (RESULTS 27a) and the other two are context for attribution, not
competing analyses of the same question -- but because reporting the best of
three recipes WOULD be multiplicity, the script also prints the across-recipe
correction (m = 36) beside it, and any sentence that compares recipes must
quote that column.  Both are printed; neither is hidden.

Run:
    python experiments/analyse_mismatch_final.py
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

# recipe -> (full tag, fullnsT tag, file stem prefix, human label)
RECIPES = [
    ("m30_pooled", "full_tc_m30", "fullnsT_tc_m30", "protect_foreign_both",
     "margin 30 / floor 15, POOLED partials"),
    ("m30_seg", "full_tc_m30_seg", "fullnsT_tc_m30_seg", "protect_foreign_seg",
     "margin 30 / floor 15, SEGMENT partials  <- the shipping recipe"),
    ("m20_seg", "full_tc_m20", "fullnsT_tc_m20", "protect_foreign_both",
     "margin 20 / floor 20, SEGMENT partials"),
]
BLOCKS = [("DiffWave-LibriSpeech", 1, "libri_t1"),
          ("DiffWave-LibriSpeech", 3, "libri_t3"),
          ("DiffWave-SC09", 1, "sc09_t1"),
          ("DiffWave-SC09", 3, "sc09_t3")]
FAMILY = 12


def speaker_of(df):
    return df.file.astype(str).str.extract(r"^spk(\d+)_")[0].astype(int)


def cells_for(full_tag, nst_tag, prefix):
    out = []
    for pur, t, stem in BLOCKS:
        p = os.path.join(R, f"{prefix}_{stem}.csv")
        if not os.path.isfile(p):
            return None, f"{prefix}_{stem}.csv"
        d = pd.read_csv(p)
        if "margin" not in d.columns:
            d["margin"] = d.cos_after_purify - d.threshold
        d["speaker"] = speaker_of(d)
        for enc in ENC_ORDER:
            sub = d[(d.encoder == enc) & (d.variant.isin([full_tag, nst_tag]))]
            piv = sub.pivot_table(index="speaker", columns="variant",
                                  values="margin")
            if full_tag not in piv.columns or nst_tag not in piv.columns:
                return None, f"{prefix}_{stem}.csv:{enc}"
            s = piv[[nst_tag, full_tag]].dropna()
            dd = (s[nst_tag] - s[full_tag]).to_numpy()
            lo, hi, _, _ = bca_ci(dd, alpha=0.05, n_boot=2000)
            out.append(dict(purifier=pur, t=t, encoder=enc, n=len(dd),
                            delta=float(dd.mean()), lo=lo, hi=hi,
                            p=wilcoxon_p(dd), helped=int((dd < 0).sum())))
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "mismatch_final.csv"))
    a = ap.parse_args()

    tables, missing = {}, {}
    for key, ft, nt, pre, _label in RECIPES:
        cells, miss = cells_for(ft, nt, pre)
        if cells is None:
            missing[key] = miss
        else:
            tables[key] = cells

    if missing:
        for k, v in missing.items():
            print(f"[final] MISSING {k}: {v}")
    if "m30_seg" not in tables:
        sys.exit("[final] the shipping recipe is absent; nothing to report")

    # within-recipe Holm (m = 12), and across-recipe Holm (m = 36)
    all_p, index = [], []
    for key, cells in tables.items():
        for i, c in enumerate(cells):
            all_p.append(c["p"])
            index.append((key, i))
    across = holm(all_p)
    for (key, i), pa in zip(index, across):
        tables[key][i]["p_holm_across"] = pa
    for key, cells in tables.items():
        praw = [c["p"] for c in cells]
        if len(praw) < FAMILY:
            praw = praw + [1.0] * (FAMILY - len(praw))
        for c, pa in zip(cells, holm(praw)[:len(cells)]):
            c["p_holm"] = pa
            c["sig"] = bool(pa < 0.05 and (c["lo"] > 0 or c["hi"] < 0))

    rows = []
    for key, _ft, _nt, _pre, label in RECIPES:
        if key not in tables:
            continue
        cells = tables[key]
        print("=" * 108)
        print(f"{label}     (n = {cells[0]['n']} speakers)")
        print("=" * 108)
        print(f"  {'purifier':22s} {'t':>2} {'encoder':22s} {'delta':>9s} "
              f"{'95% CI':>20s} {'k/n':>7s} {'p(m=12)':>9s} {'p(m=36)':>9s} sig")
        for c in cells:
            print(f"  {c['purifier']:22s} {c['t']:2d} {c['encoder']:22s} "
                  f"{c['delta']:+9.4f} [{c['lo']:+8.4f},{c['hi']:+8.4f}] "
                  f"{c['helped']:3d}/{c['n']:<3d} {c['p_holm']:9.2e} "
                  f"{c['p_holm_across']:9.2e} {'*' if c['sig'] else ''}")
            rows.append(dict(recipe=key, **c))
        k = sum(1 for c in cells if c["sig"])
        print(f"  -> {k} of {len(cells)} cells significant within this recipe "
              f"(m = 12)\n")

    pd.DataFrame(rows).to_csv(a.out, index=False)

    # ------------------------------------------------------- attribution
    def grab(key):
        return {(c["purifier"], c["t"], c["encoder"]): c["delta"]
                for c in tables.get(key, [])}

    seg, pooled, m20 = grab("m30_seg"), grab("m30_pooled"), grab("m20_seg")

    def contrast(a_map, b_map, a_name, b_name, factor):
        keys = sorted(set(a_map) & set(b_map))
        if not keys:
            print(f"  [{factor}] not available")
            return
        print("=" * 108)
        print(f"{factor}:  {a_name}  ->  {b_name}")
        print("  (difference of paired means per cell; NOT a paired test -- "
              "different crafted files)")
        print("=" * 108)
        print(f"  {'purifier':22s} {'t':>2} {'encoder':22s} "
              f"{a_name:>12s} {b_name:>12s} {'change':>9s} {'flips?':>8s}")
        flips = 0
        for k in keys:
            d = b_map[k] - a_map[k]
            fl = "SIGN" if np.sign(a_map[k]) != np.sign(b_map[k]) else ""
            flips += bool(fl)
            print(f"  {k[0]:22s} {k[1]:2d} {k[2]:22s} "
                  f"{a_map[k]:+12.4f} {b_map[k]:+12.4f} {d:+9.4f} {fl:>8s}")
        ch = np.array([b_map[k] - a_map[k] for k in keys])
        print(f"  -> mean |change| {np.abs(ch).mean():.4f}, "
              f"max |change| {np.abs(ch).max():.4f}, {flips} sign flips\n")

    if pooled and seg:
        contrast(pooled, seg, "m30pool", "m30seg",
                 "A. CRAFTING MODE, budget held at margin 30 / floor 15")
    if m20 and seg:
        contrast(m20, seg, "m20seg", "m30seg",
                 "B. BUDGET, crafting held at segment partials")

    print("=" * 108)
    print("WHAT THIS SETTLES")
    print("=" * 108)
    print("RESULTS 25 flagged a rebase that moved budget AND crafting at once,")
    print("and 26c could not say which one mattered. With the off-diagonal cell")
    print("on disk, A and B each move one factor, so the attribution is")
    print("available and the paper no longer has to hedge it.")
    print()
    print("The fourth corner (m20 + pooled) is not on disk and is not")
    print("scheduled: it buys only an interaction term, and no claim rests on")
    print("one. Recorded as a deliberate absence, not an oversight.")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
