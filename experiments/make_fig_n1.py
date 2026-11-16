"""
experiments/make_fig_n1.py
==========================
The defence paper's Figure 1 (DEFENSE_BRIEF I.6): the N1 result in one picture.

It has to show BOTH halves of RESULTS 59 at once, because the claim is the
contrast between them:

  left panel  MATCHED  -- every encoder in the crafting ensemble. Plain and
                          purifier-aware land on top of each other (0/9 cells,
                          largest |delta| 0.0232 against a 0.0247 noise floor).
  right panel HELD-OUT -- d-vector out of the ensemble. Purifier-aware pulls
                          clearly below plain (7/9 cells, ECAPA 27.6 -> 10.3 %).

Clone success, not margin, is on the y axis: it is the quantity a reader cares
about and it makes the saturation visible. Cells where BOTH protected arms sit
under 10 % are hatched, because a null there is no room rather than no effect
(RESULTS 59b) and a bar chart would otherwise hide that.

Channel is `ours` (mel DiffPure v7) by default -- the purifier the claim is
about. --purifier switches it.

Run:
  python experiments/make_fig_n1.py
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

R = os.path.join(PROJ, "experiments", "results")
OUT = os.path.join(PROJ, "paper", "fig_n1_coverage.png")
FLOOR_PCT = 10.0

PANELS = [
    dict(title="Encoder in the ensemble", file="cloning_tc_m30_deploy.csv",
         plain="plain_m30_f15_all", eot="fullnsT_m30_f15_all"),
    dict(title="Encoder held out", file="cloning_tc_m30_nodvec.csv",
         plain="plain_m30_f15_nodvec", eot="fullnsT_m30_f15_nodvec"),
]
ENCS = [("ECAPA-TDNN", "ECAPA"),
        ("WavLM-SV", "WavLM-SV"),
        ("Resemblyzer d-vector", "d-vector")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purifier", default="ours")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as e:                                  # pragma: no cover
        print(f"[fig] matplotlib unavailable ({e}); nothing written")
        return

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200,
    })
    # Colour = ARM, the same in both panels. The panels differ by ensemble
    # membership, which is the x-axis story and the titles, never by colour.
    C_UNP, C_PLAIN, C_EOT = "#9a9a9a", "#c0623a", "#1b3a6b"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    rows = []
    for ax, pan in zip(axes, PANELS):
        d = pd.read_csv(os.path.join(R, pan["file"]))
        d = d[d.purifier == args.purifier]
        x = np.arange(len(ENCS))
        w = 0.26
        series = [("unprotected", C_UNP, "no protection", -w),
                  (pan["plain"], C_PLAIN, "plain crafting", 0.0),
                  (pan["eot"], C_EOT, "purifier in the loop", +w)]
        for variant, colour, label, off in series:
            vals, ns = [], []
            for enc, _ in ENCS:
                s = d[(d.encoder == enc) & (d.variant == variant)]
                vals.append(100.0 * s.clone_succeeds.mean() if len(s) else np.nan)
                ns.append(len(s))
            ax.bar(x + off, vals, w, color=colour, label=label,
                   edgecolor="white", linewidth=0.5)
            for xi, v in zip(x + off, vals):
                if np.isfinite(v):
                    ax.text(xi, v + 1.6, f"{v:.0f}", ha="center", fontsize=7,
                            color="#333")
            rows += [dict(panel=pan["title"], variant=variant, encoder=e[0],
                          success_pct=v, n=n)
                     for e, v, n in zip(ENCS, vals, ns)]

        # Hatch the encoders where BOTH protected arms are at the floor: the
        # contrast is not identifiable there and the picture must say so.
        for xi, (enc, _) in zip(x, ENCS):
            a = d[(d.encoder == enc) & (d.variant == pan["plain"])]
            b = d[(d.encoder == enc) & (d.variant == pan["eot"])]
            if len(a) and len(b) and (100 * a.clone_succeeds.mean() < FLOOR_PCT) \
                    and (100 * b.clone_succeeds.mean() < FLOOR_PCT):
                ax.axvspan(xi - 0.42, xi + 0.42, color="#000000", alpha=0.045,
                           zorder=0, hatch="///", linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in ENCS])
        ax.set_title(pan["title"])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("clone success (%)")
    axes[0].legend(frameon=False, loc="upper left")
    h = [Patch(facecolor="#000000", alpha=0.045, hatch="///",
               label=f"both protected arms < {FLOOR_PCT:.0f} %:\nnot identifiable")]
    axes[1].legend(handles=h, frameon=False, loc="upper left")

    fig.suptitle("Putting the purifier in the crafting loop buys coverage, "
                 "not strength", fontsize=10, y=1.02)
    fig.text(0.5, -0.07,
             f"DiffVC end-to-end through {args.purifier}; 29 cross-session "
             f"test-clean speakers; budget m30_f15 (the purifier-aware arm "
             f"spends 2.9 dB more of it, §59f).",
             ha="center", fontsize=7.2, color="#333")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    csv = args.out.rsplit(".", 1)[0] + ".csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"[fig] {args.out}")
    print(f"[fig] {csv}")
    print(pd.DataFrame(rows).pivot_table(index=["panel", "encoder"],
                                         columns="variant",
                                         values="success_pct").round(1).to_string())


if __name__ == "__main__":
    main()
