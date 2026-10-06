"""
experiments/analyse_extra_encoders.py
=====================================
Decide council #13b against `paper/PREREG_13b_prospective_encoders.md`.

This script chooses nothing. The 8 cells, the endpoint, the estimator, the
threshold α* = 0.4255, the sign rule and the three outcomes were all fixed
before α was measured, and the per-encoder prediction was committed as §7 of
that document before a single crafted file was scored through either encoder.
What happens here is arithmetic plus a lookup.

THE RULE, repeated so it cannot drift (prereg §3.2):
    predict the constraint HELPS (Δ < 0) if α_e > 0.4255
    predict it REVERSES      (Δ > 0) if α_e < 0.4255
    where α_e is the encoder's survival coefficient at t = 50.

THE OUTCOME (prereg §4):
    Δ = mean paired per-speaker cos(clean, fullnsT) − cos(clean, full)
    Holm over the 8 cells; significance needs CI excluding 0 AND p_holm < .05.

Run:
  python experiments/analyse_extra_encoders.py
"""
import argparse
import io
import os
import re
import sys

# The project's UTF-8 stdout guard: this script prints greek alpha, and a
# cp1252 console cannot encode it. tests/conftest.py marks the stream so the
# rewrap is skipped under pytest capture.
if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.analyse_matched_control import bca_ci, wilcoxon_p, holm

R = os.path.join(PROJ, "experiments", "results")
PREREG = os.path.join(PROJ, "paper", "PREREG_13b_prospective_encoders.md")
ALPHA_STAR = 0.4255          # fixed in prereg §3.2; may not move
FULL, NST = "full_tc_m30_seg", "fullnsT_tc_m30_seg"


def read_alpha(path=os.path.join(R, "encoder_subspace_13b.csv"), t=50):
    if not os.path.isfile(path):
        return {}
    d = pd.read_csv(path)
    col = f"alpha_t{t}"
    if col not in d.columns:
        return {}
    m = d.groupby("direction")[col].mean()
    return {k: float(v) for k, v in m.items()}


def predicted_sign(alpha):
    """The rule. One threshold, no free parameters, fixed before measurement."""
    return "helps (negative)" if alpha > ALPHA_STAR else "reverses (positive)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(R, "extra_encoder_protection.csv"))
    ap.add_argument("--alpha-in", dest="alpha_in",
                    default=os.path.join(R, "encoder_subspace_13b.csv"))
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(R, "extra_encoder_cells.csv"))
    a = ap.parse_args()

    alphas = read_alpha(a.alpha_in)
    if not alphas:
        raise SystemExit(f"[13b] no α measurements in {a.alpha_in}")
    if not os.path.isfile(a.inp):
        print(f"[13b] NOT READY -- {os.path.basename(a.inp)} is missing.")
        print("[13b] Run experiments/eval_extra_encoders.py first, and only")
        print("      after the §7 prediction has been committed.")
        return

    # Check the committed prediction is actually in the pre-registration, so
    # this cannot be run against a document that never recorded one.
    has_pred = False
    if os.path.isfile(PREREG):
        txt = open(PREREG, encoding="utf-8").read()
        has_pred = bool(re.search(r"^##\s*7\.", txt, re.M))
    if not has_pred:
        raise SystemExit(
            "[13b] REFUSING: PREREG_13b has no §7 -- the per-encoder "
            "prediction derived from α was never committed. A prospective "
            "test whose prediction is written after the outcome is not one.")

    df = pd.read_csv(a.inp)
    cells, ps = [], []
    for enc in sorted(df.encoder.unique()):
        al = alphas.get(enc)
        if al is None:
            print(f"[13b] no α for {enc}; skipping")
            continue
        pred = predicted_sign(al)
        for (pur, t), sub in df[df.encoder == enc].groupby(["purifier", "t"]):
            piv = sub.pivot_table(index="speaker", columns="variant",
                                  values="cos_to_clean")
            if FULL not in piv.columns or NST not in piv.columns:
                continue
            s = piv[[NST, FULL]].dropna()
            d = (s[NST] - s[FULL]).to_numpy()
            lo, hi, _, _ = bca_ci(d, alpha=a.alpha, n_boot=a.n_boot)
            p = wilcoxon_p(d)
            cells.append(dict(encoder=enc, alpha_t50=al, predicted=pred,
                              purifier=pur, t=int(t), n=len(d),
                              delta=float(d.mean()), lo=lo, hi=hi, p=p,
                              helped=int((d < 0).sum())))
            ps.append(p)
    if not cells:
        raise SystemExit("[13b] no cells")
    for c, padj in zip(cells, holm(ps)):
        c["p_holm"] = padj
        c["sig"] = bool((c["hi"] < 0 or c["lo"] > 0) and padj < a.alpha)
        c["sign_matches"] = bool(
            (c["delta"] < 0) == c["predicted"].startswith("helps"))
    out = pd.DataFrame(cells)
    out.to_csv(a.out, index=False)

    print("=" * 100)
    print("#13b -- DOES α PREDICT THE CONSTRAINT'S EFFECT ON UNSEEN ENCODERS?")
    print(f"rule (fixed before measurement): α > {ALPHA_STAR} -> helps, "
          f"α < {ALPHA_STAR} -> reverses")
    print("=" * 100)
    for enc in sorted({c["encoder"] for c in cells}):
        sub = [c for c in cells if c["encoder"] == enc]
        al = sub[0]["alpha_t50"]
        print(f"\n  {enc}:  α(t=50) = {al:.4f}  ->  PREDICTED "
              f"{sub[0]['predicted'].upper()}")
        print(f"    {'purifier':22s} {'t':>2s} {'delta':>9s} {'95% CI':>20s} "
              f"{'k/n':>7s} {'p_holm':>9s}  match")
        for c in sub:
            print(f"    {c['purifier']:22s} {c['t']:2d} {c['delta']:+9.4f} "
                  f"[{c['lo']:+8.4f},{c['hi']:+8.4f}] "
                  f"{c['helped']:3d}/{c['n']:<3d} {c['p_holm']:9.2e}  "
                  f"{'YES' if c['sign_matches'] else 'no '}"
                  f"{' *' if c['sig'] else ''}")

    # ------------------------------------------------- the pre-registered read
    print("\n" + "=" * 100)
    print("VERDICT, against PREREG_13b §5")
    print("=" * 100)
    by_enc = {}
    for enc in sorted({c["encoder"] for c in cells}):
        sub = [c for c in cells if c["encoder"] == enc]
        n_match = sum(1 for c in sub if c["sign_matches"])
        t1 = [c for c in sub if c["t"] == 1]
        t1_match = all(c["sign_matches"] for c in t1) if t1 else False
        by_enc[enc] = (n_match, len(sub), t1_match)
        print(f"  {enc:12s} sign matches in {n_match}/{len(sub)} cells; "
              f"t=1 cells agree: {'yes' if t1_match else 'NO'}")

    ok = [e for e, (m, n, t1) in by_enc.items() if m >= 3 and t1]
    half = [e for e, (m, n, t1) in by_enc.items() if m == 2]

    # PREREG_13b §7.4, committed with the prediction and BEFORE the outcome:
    #   "A null -- Delta indistinguishable from zero -- is NOT a confirmation
    #    and is read under §5's outcome B."
    # §5's outcome A keys on sign agreement alone, so sign agreement over a set
    # of cells that are all consistent with zero would print A on no effect at
    # all. §7.4 is the stricter, later-committed clarification and it governs.
    n_sig = sum(1 for c in cells if c["sig"])
    all_null = (n_sig == 0)
    print()
    print(f"  cells distinguishable from zero under Holm: {n_sig}/{len(cells)}")
    if all_null:
        print("  -> every cell is consistent with Delta = 0. PREREG §7.4: a null")
        print("     is NOT a confirmation, whatever the signs do.")
    print()
    if all_null:
        print("VERDICT: OUTCOME B -- DIRECTIONAL ONLY (null effects)")
        print("  The predicted SIGN is right in every cell, which is")
        print("  suggestive, but no cell's effect is distinguishable from zero,")
        print("  so the prediction is not confirmed by it. §7.4 fixed this")
        print("  reading before the outcome was measured.")
        print("  ACTION: report the sign agreement AND the nulls together.")
        print("  C1 keeps 'explanation'; the word 'predicted' stays banned by")
        print("  FRAME_v6 §4. NO post-hoc adjustment of alpha*.")
    elif len(ok) == len(by_enc) and by_enc:
        print("VERDICT: OUTCOME A -- C1 PREDICTS OUT OF SAMPLE")
        print("  Both encoders match the rule in >=3 of 4 cells and the t=1")
        print("  cells agree.")
        print("  ACTION: C1 becomes a PREDICTION, not merely an explanation.")
        print("  FRAME_v6 §4's ban on the word 'predicted' is lifted FOR THIS")
        print("  CLAIM ONLY, citing the rule and PREREG_13b.")
    elif len(ok) == 1 or half:
        print("VERDICT: OUTCOME B -- MIXED")
        print(f"  Agreed: {', '.join(ok) if ok else 'neither'}")
        print("  ACTION: report both, naming which agreed and which did not.")
        print("  C1 keeps 'explanation'. NO post-hoc adjustment of α*.")
    else:
        print("VERDICT: OUTCOME C -- C1 DOES NOT PREDICT")
        print("  ACTION: report it as a failed prospective test. FRAME_v6 §6")
        print("  pre-holds this branch. C1 is demoted to a description of the")
        print("  three encoders it was measured on; the ordering claim stays,")
        print("  the mechanism language goes. NO re-run with a different α*, a")
        print("  different timestep, or a third encoder.")
    print("=" * 100)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
