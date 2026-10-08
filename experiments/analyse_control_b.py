"""
experiments/analyse_control_b.py
================================
Control B's verdict, read strictly off `PREREG_council5_controls.md` §B.3.

This script does not choose anything. The three branches, their thresholds and
their consequences were fixed on 2026-10-07, before the control was run. What
happens here is arithmetic plus a lookup.

    alpha_u = mean alpha at t = 50 of the UNTRAINED CAM++ directions,
              averaged over the three random inits (pre-declared primary).

    B-1   alpha_u <= 0.63   -> C1's title is wrong. "Speaker" comes out.
    B-2   alpha_u >= 0.70   -> C1 survives and is strengthened.
    B-3   0.63 < alpha_u < 0.70 -> report the number, weaken the claim.

Reference points on the same cohort and purifier: trained encoders 0.388-0.631,
`random` 0.714, `pca_span` 0.844.

Run:
    python experiments/analyse_control_b.py
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
SRC = os.path.join(R, "encoder_subspace_controlb.csv")
B1_MAX, B2_MIN = 0.63, 0.70          # prereg B.3, fixed before the run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(R, "control_b_cells.csv"))
    a = ap.parse_args()

    if not os.path.isfile(a.src):
        sys.exit(f"[ctrlB] not ready: {a.src} missing. "
                 f"Run experiments/run_control_b.sh first.")
    d = pd.read_csv(a.src)
    unt = sorted(x for x in d.direction.unique()
                 if str(x).startswith("CAM++ (untrained"))
    if not unt:
        sys.exit("[ctrlB] no untrained directions in the file")

    nspk = d.speaker.nunique()
    print("=" * 96)
    print(f"CONTROL B -- does 'speaker' do any work?   n = {nspk} speakers, "
          f"{len(unt)} random inits")
    print("Branches fixed in PREREG_council5_controls.md B.3 on 2026-10-07.")
    print("=" * 96)

    rows = []
    for col, tlab in [("alpha_t50", "t = 50"), ("alpha_t200", "t = 200")]:
        rnd = float(d[d.direction == "random"][col].mean())
        pca = float(d[d.direction == "pca_span"][col].mean())
        print(f"\n  {tlab}    random {rnd:.3f}    pca_span {pca:.3f}")
        print(f"  {'direction':34s} {'alpha':>7s} {'vs own ctl':>11s} "
              f"{'95% CI':>20s} {'p':>9s}")
        per_init = []
        for name in unt:
            e = d[d.direction == name].set_index("speaker")[col]
            c = d[d.direction == f"band_matched:{name}"].set_index("speaker")[col]
            j = pd.concat([e.rename("e"), c.rename("c")], axis=1).dropna()
            dd = (j["e"] - j["c"]).to_numpy()
            lo, hi, _, _ = bca_ci(dd, alpha=0.05, n_boot=4000)
            print(f"  {name:34s} {e.mean():7.3f} {dd.mean():+11.3f} "
                  f"[{lo:+8.3f},{hi:+8.3f}] {wilcoxon_p(dd):9.2e}")
            per_init.append(float(e.mean()))
            rows.append(dict(timestep=tlab, direction=name, alpha=float(e.mean()),
                             delta_own_control=float(dd.mean()), lo=lo, hi=hi,
                             p=wilcoxon_p(dd), n=len(dd)))
        au = float(np.mean(per_init))
        print(f"  {'MEAN over inits (primary)':34s} {au:7.3f}   "
              f"spread [{min(per_init):.3f}, {max(per_init):.3f}]")
        rows.append(dict(timestep=tlab, direction="MEAN_untrained", alpha=au,
                         delta_own_control=np.nan, lo=min(per_init),
                         hi=max(per_init), p=np.nan, n=len(per_init)))

        trained = [x for x in d.direction.unique()
                   if not str(x).startswith(("band_matched", "pca_span",
                                             "random", "CAM++ (untrained"))]
        tvals = {t: float(d[d.direction == t][col].mean()) for t in trained}
        print(f"  trained encoders: "
              + ", ".join(f"{k} {v:.3f}" for k, v in sorted(tvals.items(),
                                                            key=lambda z: z[1])))

    pd.DataFrame(rows).to_csv(a.out, index=False)

    # ----------------------------------------------------------- the verdict
    au50 = float(np.mean([d[d.direction == n]["alpha_t50"].mean()
                          for n in unt]))
    print("\n" + "=" * 96)
    print("VERDICT -- read off B.3, not chosen now")
    print("=" * 96)
    print(f"  alpha_untrained at t = 50 (mean over {len(unt)} inits) = "
          f"{au50:.4f}")
    print(f"  thresholds: B-1 if <= {B1_MAX:.2f}, B-2 if >= {B2_MIN:.2f}, "
          f"else B-3\n")
    if au50 <= B1_MAX:
        print("  *** B-1 FIRES. C1's TITLE IS WRONG. ***")
        print("  The untrained network of the same architecture is ALSO")
        print("  purifier-hostile, so the effect is about gradients of deep")
        print("  speech networks through a denoiser, not about speaker")
        print("  identity. Per the pre-registration and without discretion:")
        print("    - 'Speaker' comes out of the title (FRAME_v7 2 holds the")
        print("      fallback: 'What a Speech Purifier Erases, and What That")
        print("      Does to the Instruments').")
        print("    - C1 is demoted; the paper leads with C2, the measurement")
        print("      contribution.")
        print("    - B.4: the honest statement is the DISJUNCTION 'identity or")
        print("      architecture', never 'architecture'.")
    elif au50 >= B2_MIN:
        print("  *** B-2 FIRES. C1 SURVIVES AND IS STRENGTHENED. ***")
        print("  An untrained network of the same architecture is NOT")
        print("  purifier-hostile, so the effect tracks LEARNED identity")
        print("  rather than architecture or gradient roughness. This is the")
        print("  control that earns the word 'speaker'. It belongs in Fig. 1.")
    else:
        print("  *** B-3 FIRES. Report the number, weaken the claim. ***")
        print("  Write: 'learned speaker encoders are more purifier-hostile")
        print(f"  than an untrained network of the same architecture, by "
              f"{au50 - min(0.388, 0.631):.3f}-ish', with NO mechanism")
        print("  language. Do not re-run at other settings (B.4).")

    print("\n  CAVEAT TO STATE WHICHEVER BRANCH FIRED (recorded before the run):")
    print("  the untrained network's embeddings are nearly degenerate")
    print("  (same-speaker cosine ~0.995, different-speaker ~0.975-0.990), so")
    print("  its gradient is the direction that breaks a near-constant")
    print("  function. That is not obviously the same KIND of object as a")
    print("  trained encoder's identity direction, which weakens B-2 as")
    print("  evidence and is a limitation of the control, not of the result.")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
