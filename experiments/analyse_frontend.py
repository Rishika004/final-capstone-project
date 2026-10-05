"""
experiments/analyse_frontend.py
===============================
Decide council #10 from `frontend_rescore.csv`, against the branches written
into `eval_frontend.py` before the run.

THE QUESTION
------------
R1: the d-vector is the encoder that fails under purification, and it is also
the only one scored through a front end the optimiser never differentiated
through (`preprocess_wav`: volume normalise + VAD trim, then partial
averaging). So is the per-encoder pattern a property of the encoders, or an
artefact of who is scored through what?

THE TEST
--------
Each encoder is re-scored under the other kind of front end, so the swap runs
in both directions:

    WavLM-SV / ECAPA   native = raw waveform       swapped = + preprocess_wav
    d-vector           native = preprocess_wav     swapped = no preprocess_wav

If the confound were real, giving WavLM and ECAPA the d-vector's front end
should cost them much of their protection, and taking it away from the
d-vector should restore some of its own.

ESTIMATOR, fixed before the run (PLAN_V3 §4)
--------------------------------------------
  unit         speaker
  quantity     cos(clean original, arm) under front end f, both sides through
               the same front end
  statistic    mean paired difference, swapped minus native, per
               (encoder, arm, purifier)
  interval     BCa bootstrap 95 % CI, 2000 resamples
  p            Wilcoxon signed-rank, two-sided, Holm within each encoder
  sign         a POSITIVE difference on a protected arm means the swap moved
               the arm back TOWARDS the speaker, i.e. less protection

The reading is comparative, not a bare threshold: what matters is whether the
shift on WavLM/ECAPA is large beside the d-vector's own native-vs-swapped
shift, and beside the protection effect itself.

Run:
  python experiments/analyse_frontend.py
  python experiments/analyse_frontend.py --purifier diffwave
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
from utils.csvio import read_csv

DEFAULT_IN = os.path.join(PROJ, "experiments", "results",
                          "frontend_rescore.csv")
SHIP = "fullnsT_tc_m30_seg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--purifier", default="", help="limit to one purifier")
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--out", default=os.path.join(
        PROJ, "experiments", "results", "frontend_summary.csv"))
    a = ap.parse_args()

    if not os.path.isfile(a.inp):
        raise SystemExit(f"not found: {a.inp} -- run experiments/eval_frontend.py")
    df = read_csv(a.inp)
    if a.purifier:
        df = df[df.purifier == a.purifier]
    purs = list(dict.fromkeys(df.purifier))
    fes = [f for f in dict.fromkeys(df.front_end) if f != "native"]

    print("=" * 92)
    print("COUNCIL #10 -- the front-end confound")
    print(f"{df.speaker.nunique()} speakers | purifiers: {', '.join(purs)} | "
          f"front ends vs native: {', '.join(fes)}")
    shas = sorted(set(df["code_sha"].dropna().astype(str))) if "code_sha" in df else []
    if shas:
        print(f"code_sha: {', '.join(shas)}")
    print("=" * 92)

    rows = []
    for pur in purs:
        d = df[df.purifier == pur]
        print(f"\n{'=' * 92}\npurifier = {pur}\n{'=' * 92}")

        # Context first: how big is the protection effect at all, natively?
        print("\n  protection, native front end: cos(clean, arm) -- "
              "lower is more protection")
        base = d[d.front_end == "native"].pivot_table(
            index="speaker", columns=["encoder", "variant"],
            values="cos_to_clean")
        for enc in sorted({c[0] for c in base.columns}):
            bits = []
            for arm in sorted({c[1] for c in base.columns if c[0] == enc}):
                bits.append(f"{arm}={base[(enc, arm)].mean():+.3f}")
            print(f"    {enc:22s} " + "  ".join(bits))

        for enc in sorted(d.encoder.unique()):
            sub = d[d.encoder == enc]
            print(f"\n  --- {enc} ---")
            print(f"    {'front end':9s} {'arm':22s} {'delta':>8s} "
                  f"{'95% CI':>18s} {'n':>4s} {'p_holm':>9s}")
            cells, ps = [], []
            for fe in fes:
                for arm in sorted(sub.variant.unique()):
                    piv = sub[sub.variant == arm].pivot_table(
                        index="speaker", columns="front_end",
                        values="cos_to_clean")
                    if fe not in piv.columns or "native" not in piv.columns:
                        continue
                    s = piv[[fe, "native"]].dropna()
                    dd = (s[fe] - s["native"]).to_numpy()
                    lo, hi, _, _ = bca_ci(dd, n_boot=a.n_boot)
                    p = wilcoxon_p(dd)
                    cells.append(dict(purifier=pur, encoder=enc, front_end=fe,
                                      variant=arm, n=len(dd),
                                      delta=float(dd.mean()), lo=lo, hi=hi,
                                      p=p))
                    ps.append(p)
            for c, padj in zip(cells, holm(ps)):
                c["p_holm"] = padj
                print(f"    {c['front_end']:9s} {c['variant']:22s} "
                      f"{c['delta']:+8.4f} [{c['lo']:+7.4f},{c['hi']:+7.4f}] "
                      f"{c['n']:4d} {padj:9.2e}")
            rows.extend(cells)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)

    # ------------------------------------------------- the pre-registered read
    print("\n" + "=" * 92)
    print("VERDICT, against the branches in eval_frontend.py")
    print("=" * 92)
    swap = out[(out.front_end == "swapped") & (out.variant == SHIP)]
    if not len(swap):
        print("  no swapped rows for the shipping arm; cannot decide")
        return
    dv = swap[swap.encoder.str.contains("d-vector")]
    others = swap[~swap.encoder.str.contains("d-vector")]
    for pur in purs:
        o = others[others.purifier == pur]
        v = dv[dv.purifier == pur]
        if not len(o) or not len(v):
            continue
        worst = o.delta.abs().max()
        print(f"\n  purifier={pur}, shipping arm:")
        for _, r in o.iterrows():
            print(f"    {r.encoder:22s} swapped-native {r.delta:+.4f} "
                  f"[{r.lo:+.4f},{r.hi:+.4f}]")
        for _, r in v.iterrows():
            print(f"    {r.encoder:22s} swapped-native {r.delta:+.4f} "
                  f"[{r.lo:+.4f},{r.hi:+.4f}]  (front end REMOVED)")
        dvd = float(v.delta.iloc[0])
        print(f"    largest |shift| among WavLM/ECAPA: {worst:.4f}; "
              f"d-vector's own shift: {abs(dvd):.4f}")
        if abs(dvd) > 0.05:
            print("    -> the d-vector moves materially when its front end is "
                  "removed. Branch 3: the front end WAS doing work, contrary "
                  "to §38's 0.985. Reinstating 2.4 has to be reconsidered.")
        elif worst < 0.10:
            print("    -> neither direction of the swap moves the shipping arm "
                  "much. Branch 1: the CONFOUND IS DEAD -- the per-encoder "
                  "pattern is a property of the encoders, not of who is scored "
                  "through what.")
        else:
            print("    -> WavLM/ECAPA lose materially under the d-vector's "
                  "front end. Branch 2: the confound is REAL and the "
                  "per-encoder ordering must be restated with a matched front "
                  "end.")
    print("\n  The resample rows (rs+2 / rs-2) are a separate question: they "
          "test\n  robustness to a benign channel nobody crafted against, not "
          "the asymmetry.")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
