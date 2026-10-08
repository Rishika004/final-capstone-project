"""
experiments/audit_summarise_rms.py
==================================
Tables for RESULTS section 58e from `audit_rms_sensitivity.py`'s output:
alpha against scale per direction, the ENCODER-MINUS-CONTROL CONTRAST against
scale (the quantity every alpha verdict is read from), and the sign asymmetry
alpha(+d) - alpha(-d) at the protocol scale.

    python experiments/audit_summarise_rms.py
"""
import os

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(PROJ, "experiments", "results")


def main():
    d = pd.read_csv(os.path.join(R, "audit_rms_sensitivity.csv"))
    for part, g in d.groupby("part"):
        spk = sorted(g.speaker.unique())
        print("=" * 100)
        print(f"{part}   speakers {spk}")
        print("=" * 100)
        gp = g[g.sign == 1]
        piv = gp.pivot_table(index="direction", columns="scale", values="alpha", aggfunc="mean")
        print("\nmean alpha by scale (sign +1):")
        print(piv.round(3).to_string())
        pk = gp.pivot_table(index="direction", columns="scale", values="peak", aggfunc="mean")
        print("\nmean peak |d| by scale (local amplitude; rms is the column header):")
        print(pk.round(3).to_string())
        # contrast vs scale, paired per speaker
        print("\nCONTRAST alpha(target) - alpha(band_matched:target), mean over speakers, by scale:")
        tg = [c for c in piv.index if f"band_matched:{c}" in piv.index]
        per = gp.pivot_table(index=["speaker", "direction"], columns="scale", values="alpha")
        for t in tg:
            diffs = []
            for s in spk:
                try:
                    diffs.append(per.loc[(s, t)] - per.loc[(s, f"band_matched:{t}")])
                except KeyError:
                    pass
            if diffs:
                m = pd.concat(diffs, axis=1).T
                print(f"  {t[:40]:40s} " + "  ".join(f"s={c:.3g}: {m[c].mean():+.3f} "
                      f"({int((m[c] < 0).sum())}/{len(m)} below)" for c in m.columns))
        # contrast vs random too
        print("\nCONTRAST alpha(direction) - alpha(random), mean over speakers, by scale:")
        for t in [c for c in piv.index if c != "random" and not c.startswith("band_matched")]:
            diffs = []
            for s in spk:
                try:
                    diffs.append(per.loc[(s, t)] - per.loc[(s, "random")])
                except KeyError:
                    pass
            if diffs:
                m = pd.concat(diffs, axis=1).T
                print(f"  {t[:40]:40s} " + "  ".join(f"s={c:.3g}: {m[c].mean():+.3f}" for c in m.columns))
        s0 = 0.2 if part.startswith("A") else 0.05
        both = g[(g.scale - s0).abs() < 1e-9].pivot_table(index=["direction", "speaker"],
                                                          columns="sign", values="alpha")
        if -1 in both and 1 in both:
            asym = (both[1] - both[-1])
            print(f"\nsign asymmetry alpha(+d) - alpha(-d) at the protocol scale {s0}: "
                  f"per direction mean / max |.|")
            for t, v in asym.groupby(level=0):
                print(f"  {t[:40]:40s} {v.mean():+.4f}  max|.| {v.abs().max():.4f}   "
                      f"alpha(+d) {both[1].loc[t].mean():.3f}  alpha(-d) {both[-1].loc[t].mean():.3f}")
        print()


if __name__ == "__main__":
    main()
