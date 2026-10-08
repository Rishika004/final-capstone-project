"""
experiments/analyse_13c.py
==========================
Council #13c's verdict: does C1's level claim hold on DiffWave too?

Reads `encoder_subspace_wav.csv` and applies the reading fixed in
`run_13c.sh` before the run:

    PASS   every encoder below its OWN band-matched control, both checkpoints,
           both timesteps -> C1 is purifier-general; its last PENDING closes.
    MIXED  some cells below, some not -> report per cell; C1's claim narrows.
    FAIL   not below -> C1 is a statement about our mel purifier only, which
           weakens C1-as-explanation and belongs in the abstract.

THE LEVELS ARE NOT COMPARABLE TO THE MEL ALPHAS. DiffWave at t = 1-3 is close
to identity-plus-noise, so alpha sits near 1 here. Only the within-purifier
contrast is meaningful, and this script prints the contrast, never a
cross-purifier difference.

Run:
    python experiments/analyse_13c.py
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
SRC = os.path.join(R, "encoder_subspace_wav.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(R, "wav_alpha_cells.csv"))
    a = ap.parse_args()

    if not os.path.isfile(a.src):
        sys.exit(f"[13c] not ready: {a.src} missing. "
                 f"Run experiments/run_13c.sh first.")
    d = pd.read_csv(a.src)
    tcols = sorted(c for c in d.columns if c.startswith("alpha_t"))
    encs = [x for x in d.direction.unique()
            if not str(x).startswith(("band_matched", "random", "pca_span"))]

    print("=" * 100)
    print("#13c -- alpha through DiffWave, waveform domain")
    print(f"speakers {d.speaker.nunique()}   purifiers "
          f"{sorted(d.purifier.unique())}   {len(encs)} encoders")
    print("LEVELS ARE NOT COMPARABLE WITH THE MEL ALPHAS. Only the contrast")
    print("against each encoder's own band-matched control is read here.")
    print("=" * 100)

    rows = []
    for pur, dp in d.groupby("purifier"):
        for col in tcols:
            t = col.replace("alpha_t", "")
            rnd = float(dp[dp.direction == "random"][col].mean())
            praw, tmp = [], []
            for e in encs:
                ev = dp[dp.direction == e].set_index("speaker")[col]
                cv = dp[dp.direction == f"band_matched:{e}"] \
                    .set_index("speaker")[col]
                j = pd.concat([ev.rename("e"), cv.rename("c")], axis=1).dropna()
                if len(j) < 5:
                    continue
                dd = (j["e"] - j["c"]).to_numpy()
                lo, hi, _, _ = bca_ci(dd, alpha=0.05, n_boot=4000)
                praw.append(wilcoxon_p(dd))
                tmp.append(dict(purifier=pur, t=int(t), encoder=e, n=len(dd),
                                alpha=float(j["e"].mean()),
                                control=float(j["c"].mean()),
                                random=rnd, delta=float(dd.mean()),
                                lo=lo, hi=hi, p=praw[-1],
                                below=int((dd < 0).sum())))
            if not tmp:
                continue
            for r, pa in zip(tmp, holm(praw)):
                r["p_holm"] = pa
                r["sig_below"] = bool(pa < 0.05 and r["hi"] < 0)
            print(f"\n  {pur}, t = {t}     random {rnd:.4f}")
            print(f"  {'encoder':22s} {'alpha':>8s} {'own ctl':>9s} "
                  f"{'delta':>9s} {'95% CI':>20s} {'k/n':>7s} "
                  f"{'p_holm':>9s} below?")
            for r in tmp:
                print(f"  {r['encoder']:22s} {r['alpha']:8.4f} "
                      f"{r['control']:9.4f} {r['delta']:+9.4f} "
                      f"[{r['lo']:+8.4f},{r['hi']:+8.4f}] "
                      f"{r['below']:3d}/{r['n']:<3d} {r['p_holm']:9.2e} "
                      f"{'YES' if r['sig_below'] else 'no'}")
            rows += tmp

    if not rows:
        sys.exit("[13c] no complete cells yet")
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)

    k, tot = int(out.sig_below.sum()), len(out)
    print("\n" + "=" * 100)
    print("VERDICT -- read off run_13c.sh, fixed before the run")
    print("=" * 100)
    print(f"  {k} of {tot} cells have the encoder significantly below its own "
          f"band-matched control.")
    if k == tot:
        print("\n  *** PASS. C1's level claim is PURIFIER-GENERAL. ***")
        print("  It holds on a waveform diffusion model we did not train, in a")
        print("  different representation, on both checkpoints and both")
        print("  evaluation timesteps. The 32/37/43 measurement and the C2")
        print("  phenomenon now sit on the SAME purifier, which is the whole")
        print("  point of #13c. C1's last PENDING closes.")
    elif k == 0:
        print("\n  *** FAIL. C1 is a statement about OUR mel purifier. ***")
        print("  The mechanism that explains the DiffWave phenomenon does not")
        print("  hold on DiffWave. That substantially weakens")
        print("  C1-as-explanation and belongs in the ABSTRACT, not the")
        print("  limitations.")
    else:
        print("\n  *** MIXED. Report per cell. ***")
        print("  C1's claim narrows to 'on our mel purifier, and on DiffWave")
        print("  for the following encoders'. Weaker, honest, and the figure")
        print("  gains a panel rather than a footnote. Cells NOT below:")
        for _, r in out[~out.sig_below].iterrows():
            print(f"    {r.purifier} t={int(r.t)} {r.encoder}: "
                  f"delta {r.delta:+.4f} [{r.lo:+.4f},{r.hi:+.4f}]")
    print("\n  Both checkpoints are read TOGETHER. Picking the one that agrees")
    print("  is the 25 failure mode.")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
