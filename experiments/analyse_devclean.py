"""
experiments/analyse_devclean.py
===============================
Decide #5 — the out-of-sample replication of C2 on dev-clean — strictly against
`paper/PREREG_05_devclean_replication.md`.

This script does not choose anything. The 12 cells, the endpoint, the
estimator, the multiplicity family and the three outcomes were all fixed in the
pre-registration before the first dev-clean file was crafted. What happens here
is arithmetic plus a lookup into §6's outcome table.

ENDPOINT (prereg §4), repeated so it cannot drift:
    mean paired per-speaker difference in (cos_after_purify - threshold)
    between the fullnsT and full arms, fullnsT - full, per cell.
    unit            speaker, n = 31
    sign            NEGATIVE = the constraint improves protection
                    POSITIVE = the reversal
    interval        BCa bootstrap 95 % CI, 2000 resamples
    p               Wilcoxon signed-rank, two-sided
    multiplicity    Holm over the 12 cells of THIS run, one family
    significant     CI excluding 0 AND Holm p < 0.05, both required

The three load-bearing predictions, from §5:
    P1  WavLM negative in all four purifier x t cells
    P2  the d-vector POSITIVE at t = 1 on both purifiers
    P3  ECAPA keeps the sign in all four cells (no significance required)

Cells 6 and 12 (+0.014) and cell 4 (-0.026) are below the detectable effect at
n = 31, so their non-significance is uninformative by prior agreement and the
verdict does not depend on them.

Run:
  python experiments/analyse_devclean.py
  python experiments/analyse_devclean.py --thresholds-from devclean   # §2.2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.analyse_matched_control import bca_ci, wilcoxon_p, holm

R = os.path.join(PROJ, "experiments", "results")
FULL, NST = "full_dc_m30_seg_bpda", "fullnsT_dc_m30_seg_bpda"
ENC_ORDER = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]

# (purifier, t) -> results file stem. Fixed here; these are the only four
# blocks. `--suffix` selects the pre-specified sensitivity scoring (§2.2),
# which reads the SAME crafted files re-scored against thresholds calibrated on
# dev-clean itself. It is a robustness check on the verdict, never the result.
BLOCK_STEMS = [("DiffWave-LibriSpeech", 1, "dc5_foreign_libri_t1"),
               ("DiffWave-LibriSpeech", 3, "dc5_foreign_libri_t3"),
               ("DiffWave-SC09", 1, "dc5_foreign_sc09_t1"),
               ("DiffWave-SC09", 3, "dc5_foreign_sc09_t3")]

# The test-clean point estimates from RESULTS §27a, i.e. the predictions.
PRED = {("DiffWave-LibriSpeech", 1, "WavLM-SV"): -0.076,
        ("DiffWave-LibriSpeech", 1, "ECAPA-TDNN"): -0.046,
        ("DiffWave-LibriSpeech", 1, "Resemblyzer d-vector"): +0.071,
        ("DiffWave-LibriSpeech", 3, "WavLM-SV"): -0.026,
        ("DiffWave-LibriSpeech", 3, "ECAPA-TDNN"): -0.029,
        ("DiffWave-LibriSpeech", 3, "Resemblyzer d-vector"): +0.014,
        ("DiffWave-SC09", 1, "WavLM-SV"): -0.078,
        ("DiffWave-SC09", 1, "ECAPA-TDNN"): -0.038,
        ("DiffWave-SC09", 1, "Resemblyzer d-vector"): +0.075,
        ("DiffWave-SC09", 3, "WavLM-SV"): -0.046,
        ("DiffWave-SC09", 3, "ECAPA-TDNN"): -0.034,
        ("DiffWave-SC09", 3, "Resemblyzer d-vector"): +0.014}
# Cells the run is underpowered for (prereg §5): non-significance here is
# uninformative and must not be read as a failed replication.
UNDERPOWERED = {("DiffWave-LibriSpeech", 3, "Resemblyzer d-vector"),
                ("DiffWave-SC09", 3, "Resemblyzer d-vector"),
                ("DiffWave-LibriSpeech", 3, "WavLM-SV")}


def margin(df):
    """cos_after_purify - threshold, per row, named consistently."""
    d = df.copy()
    if "margin" not in d.columns:
        d["margin"] = d.cos_after_purify - d.threshold
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--suffix", default="",
                    help="read the sensitivity scoring instead, e.g. '_dcthr' "
                         "for the dev-clean-threshold re-scoring (prereg §2.2)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(R, f"dc5_cells{a.suffix}.csv")

    BLOCKS = [(p, t, stem + a.suffix + ".csv") for p, t, stem in BLOCK_STEMS]
    if a.suffix:
        print("=" * 100)
        print(f"SENSITIVITY SCORING ({a.suffix.lstrip('_')}) -- prereg 2.2.")
        print("This is NOT the result. If the verdict differs from the")
        print("test-clean-threshold scoring, the verdict is "
              "'threshold-dependent'")
        print("and NEITHER scoring is reported as the result.")
        print("=" * 100)
    cells, missing = [], []
    for pur, t, fname in BLOCKS:
        p = os.path.join(R, fname)
        if not os.path.isfile(p):
            missing.append(fname)
            continue
        d = margin(pd.read_csv(p))
        d["speaker"] = d.file.astype(str).str.extract(r"^spk(\d+)_")[0].astype(int)
        for enc in ENC_ORDER:
            sub = d[(d.encoder == enc) & (d.variant.isin([FULL, NST]))]
            piv = sub.pivot_table(index="speaker", columns="variant",
                                  values="margin")
            if FULL not in piv.columns or NST not in piv.columns:
                missing.append(f"{fname}:{enc}")
                continue
            s = piv[[NST, FULL]].dropna()
            dd = (s[NST] - s[FULL]).to_numpy()
            lo, hi, _, _ = bca_ci(dd, alpha=a.alpha, n_boot=a.n_boot)
            cells.append(dict(purifier=pur, t=t, encoder=enc, n=len(dd),
                              delta=float(dd.mean()), lo=lo, hi=hi,
                              p=wilcoxon_p(dd),
                              improved=int((dd < 0).sum()),
                              predicted=PRED[(pur, t, enc)],
                              underpowered=(pur, t, enc) in UNDERPOWERED))
    if missing:
        print(f"[dc5] NOT READY -- missing: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
        if not cells:
            return
    if len(cells) != 12:
        print(f"[dc5] {len(cells)} of 12 cells present. The verdict in §6 is "
              f"defined on all 12; what follows is partial and is NOT the "
              f"result.")

    # BUG FIX 2026-10-07 (council v5 code review, BUG D), two defects here.
    #
    # (1) The Holm family was sized by the number of cells PRESENT, while the
    #     pre-registration (§4) fixes it at all 12: "Holm over the 12 cells of
    #     this run -- one family, fixed here." With 6 of 12 blocks on disk
    #     every p_holm was halved and every `*` in the printed table and in
    #     dc5_cells.csv was too liberal. The §6 verdict was safe because it is
    #     gated on len(cells) == 12, but the table and the written CSV were
    #     not, and a partial table is exactly what gets glanced at mid-run.
    #     The family is now the pre-registered size regardless of how many
    #     cells are on disk, which can only make a partial table CONSERVATIVE.
    #
    # (2) `sig` was two-sided -- (hi < 0 or lo > 0) -- while the endpoint is
    #     directional and per-cell: §5 predicts a SIGN for each cell. So a cell
    #     where the constraint significantly HURT earned the same `*` as a
    #     confirmation. Significance now additionally requires the CI to
    #     exclude zero on the PREDICTED side.
    FAMILY = 12          # prereg §4, fixed before the run
    praw = [c["p"] for c in cells]
    if len(praw) < FAMILY:
        praw = praw + [1.0] * (FAMILY - len(praw))   # absent cells cannot help
    for c, padj in zip(cells, holm(praw)[:len(cells)]):
        c["p_holm"] = padj
        excludes_zero = (c["hi"] < 0) if c["predicted"] < 0 else (c["lo"] > 0)
        c["sig"] = bool(excludes_zero and padj < a.alpha)
        c["sig_wrong_way"] = bool(padj < a.alpha and not excludes_zero
                                  and (c["lo"] > 0 or c["hi"] < 0))
    if any(c.get("sig_wrong_way") for c in cells):
        print("[dc5] WARNING: at least one cell is significant in the "
              "direction OPPOSITE to its prediction. That is not a "
              "confirmation; see the sig_wrong_way column.")
    out = pd.DataFrame(cells)
    out.to_csv(a.out, index=False)

    n = out.n.iloc[0] if len(out) else 0
    print("=" * 100)
    print("#5 OUT-OF-SAMPLE REPLICATION OF C2 -- dev-clean, n = "
          f"{n} speakers, thresholds calibrated on test-clean")
    print("endpoint: paired (fullnsT - full) margin. NEGATIVE = constraint "
          "helps. Holm over the 12 cells.")
    print("=" * 100)
    print(f"  {'#':>2} {'purifier':22s} {'t':>2} {'encoder':22s} "
          f"{'pred':>7s} {'delta':>8s} {'95% CI':>19s} {'k/n':>7s} "
          f"{'p_holm':>9s}  sig")
    for i, c in enumerate(cells, 1):
        flag = "*" if c["sig"] else (" " if not c["underpowered"] else "~")
        print(f"  {i:2d} {c['purifier']:22s} {c['t']:2d} {c['encoder']:22s} "
              f"{c['predicted']:+7.3f} {c['delta']:+8.4f} "
              f"[{c['lo']:+8.4f},{c['hi']:+8.4f}] "
              f"{c['improved']:3d}/{c['n']:<3d} {c['p_holm']:9.2e}  {flag}")
    print("   * = CI excludes 0 and Holm p < .05   ~ = pre-declared "
          "underpowered, non-significance uninformative")

    # ----------------------------------------------------- the three readings
    def cell(pur, t, enc):
        for c in cells:
            if (c["purifier"], c["t"], c["encoder"]) == (pur, t, enc):
                return c
        return None

    wav = [cell(p, t, "WavLM-SV") for p, t, _ in BLOCKS]
    eca = [cell(p, t, "ECAPA-TDNN") for p, t, _ in BLOCKS]
    dv1 = [cell(p, 1, "Resemblyzer d-vector")
           for p in ("DiffWave-LibriSpeech", "DiffWave-SC09")]
    wav, eca, dv1 = ([x for x in v if x] for v in (wav, eca, dv1))

    p1_sign = sum(1 for c in wav if c["delta"] < 0)
    p1_sig = sum(1 for c in wav if c["sig"] and c["delta"] < 0)
    p2_sign = sum(1 for c in dv1 if c["delta"] > 0)
    p2_sig = sum(1 for c in dv1 if c["sig"] and c["delta"] > 0)
    p3_sign = sum(1 for c in eca if c["delta"] < 0)

    print("\n" + "-" * 100)
    print("THE THREE PRE-REGISTERED PREDICTIONS")
    print("-" * 100)
    print(f"  P1  WavLM negative in all four cells      : "
          f"{p1_sign}/4 negative, {p1_sig}/4 significant")
    print(f"  P2  d-vector POSITIVE at t=1, both purifiers: "
          f"{p2_sign}/2 positive, {p2_sig}/2 significant")
    print(f"  P3  ECAPA keeps the sign in all four cells : "
          f"{p3_sign}/4 negative (no significance required)")

    if len(cells) != 12:
        print("\n  Partial data -- no verdict.")
        return

    print("\n" + "=" * 100)
    if p1_sign < 3 or p2_sign < 2:
        print("VERDICT: OUTCOME C -- NOT REPLICATED")
        print("  P1 fails (>=2 WavLM cells non-negative) or P2 fails (a t=1")
        print("  d-vector cell is non-positive).")
        print("  ACTION, fixed in advance: C2 fails out of sample and that is")
        print("  the reported result. FRAME_v6 §6 pre-holds this branch. C2 is")
        print("  demoted to a single-sample observation about test-clean, the")
        print("  paper leads with C1 + C3, and §27 carries an explicit")
        print("  non-replication note. NO re-run at another budget, timestep,")
        print("  purifier or speaker subset, and no search for a cell where it")
        print("  clears. A null here is publishable and cheaper to report than")
        print("  to launder.")
    elif p1_sign == 4 and p1_sig >= 3 and p2_sign == 2 and p2_sig == 2 and p3_sign >= 3:
        print("VERDICT: OUTCOME A -- REPLICATED")
        print("  ACTION: C2 becomes a finding, replicated out of sample on")
        print("  disjoint speakers with disjoint thresholds. Report both")
        print("  samples side by side with their CIs; DO NOT POOL them. This")
        print("  also retires council #5 for C2.")
    else:
        print("VERDICT: OUTCOME B -- DIRECTIONAL ONLY")
        print("  Signs preserved where the run has power, significance")
        print("  substantially weaker than §27.")
        print("  ACTION: write 'directionally reproduced out of sample,")
        print("  underpowered at n = 31'. Report both samples and the per-cell")
        print("  CIs and state the power calculation. DO NOT write")
        print("  'replicated'. C2 stays a supported hypothesis, C1 keeps the")
        print("  lead, and the limitation section says the confirmatory sample")
        print("  is 29 + 31 speakers from one corpus.")
    print("=" * 100)
    if a.suffix:
        print("\nThis was the SENSITIVITY scoring. Compare its verdict with the")
        print("primary one (no --suffix). Same verdict -> the primary stands.")
        print("Different verdict -> 'threshold-dependent', and neither is")
        print("reported as the result (prereg 2.2).")
    else:
        print("\nStill required before this verdict is final (prereg 2.2 / 6):")
        print("  the sensitivity check -- the same 12 cells re-scored against")
        print("  thresholds calibrated on dev-clean itself:")
        print("    bash experiments/run_devclean_sensitivity.sh")
        print("    python experiments/analyse_devclean.py --suffix _dcthr")
        print("  If the verdict flips, it is 'threshold-dependent' and NEITHER")
        print("  scoring is the result.")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
