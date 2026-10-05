"""
experiments/make_figs_paper3.py
===============================
Figures for Paper 3. Each figure is generated only if its CSV exists, so this can
be run at any point while the experiment queue is still going.

  fig2_transfer.png    cross-purifier transfer (E2): pass-through error of each
                       direction family on the purifier it was optimised against
                       vs. an independently trained one.
  fig3_dimension.png   subspace dimension (E5): pass-through error as a function
                       of how many mutually orthogonal directions we demand.
  fig4_crossexample.png cross-example transfer (E7): R_t matrix, own vs other.

Run:
  python experiments/make_figs_paper3.py
"""
import sys, io, os

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
from utils.csvio import read_csv as read_csv_safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(PROJ, "experiments", "results")
FIG = os.path.join(PROJ, "paper")

LABEL = {"random": "random", "null": "null (zero-noise)",
         "null_t100": "null (matched, t=100)", "null_t200": "null (matched, t=200)",
         "range": "range (worst case)"}
ORDER = ["range", "random", "null", "null_t100", "null_t200"]


def _order(df, col="kind"):
    present = [k for k in ORDER if k in set(df[col])]
    return present + [k for k in sorted(set(df[col])) if k not in present]


def fig_transfer():
    f = os.path.join(R, "nullspace_xfer_v3_to_v4.csv")
    if not os.path.isfile(f):
        return print("skip fig2 (no transfer csv)")
    df = read_csv_safe(f)
    if "R_t200_eval" not in df.columns:
        return print("skip fig2 (no eval columns)")
    g = df.groupby("kind")[["R_t200", "R_t200_eval"]].mean()
    kinds = _order(df)
    g = g.loc[kinds]
    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(x - 0.2, g.R_t200, 0.4, label="purifier v3 (optimised against)",
           color="#3b6ea5")
    ax.bar(x + 0.2, g.R_t200_eval, 0.4, label="purifier v4 (never seen)",
           color="#c1662f")
    ax.axhline(g.loc["random", "R_t200"], ls="--", lw=1, c="0.4")
    ax.text(len(kinds) - 0.5, g.loc["random", "R_t200"] * 1.1, "random baseline",
            ha="right", fontsize=8, color="0.3")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(k, k) for k in kinds], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"pass-through error $R_t$  (lower = survives)")
    ax.set_title("The pass-through subspace transfers across purifiers", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG, "fig2_transfer.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


def fig_dimension():
    f = os.path.join(R, "nullspace_dimension.csv")
    if not os.path.isfile(f):
        return print("skip fig3 (no dimension csv)")
    df = read_csv_safe(f)
    g = df.groupby("k")[["R_t"]].mean()
    ref = df.R_t_random.mean()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(g.index, g.R_t, "-o", ms=3, color="#3b6ea5")
    ax.axhline(ref, ls="--", lw=1, c="0.4")
    ax.text(g.index.max(), ref * 1.05, "random direction", ha="right",
            fontsize=8, color="0.3")
    ax.set_yscale("log")
    ax.set_xlabel("number of mutually orthogonal directions $k$")
    ax.set_ylabel(r"pass-through error $R_t$")
    ax.set_title("How many directions pass through the purifier?", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG, "fig3_dimension.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


def fig_crossexample():
    f = os.path.join(R, "crossexample_transfer.csv")
    if not os.path.isfile(f):
        return print("skip fig4 (no cross-example csv)")
    df = read_csv_safe(f)
    spks = sorted(set(df.dir_from) | set(df.eval_at))
    m = np.full((len(spks), len(spks)), np.nan)
    for _, r in df.iterrows():
        m[spks.index(r.dir_from), spks.index(r.eval_at)] = r.R_t
    rnd = float(df.R_t_random.mean())
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # Scale the colour map up to the RANDOM baseline. Auto-scaling spans only the
    # matrix's own range, which makes the off-diagonal look poor when in fact every
    # cell -- diagonal and off-diagonal alike -- sits one to two orders of magnitude
    # below a random direction. That comparison is the entire point of the figure.
    im = ax.imshow(np.log10(m), cmap="viridis_r",
                   vmin=float(np.nanmin(np.log10(m))), vmax=float(np.log10(rnd)))
    ax.set_xticks(range(len(spks))); ax.set_xticklabels(spks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(spks))); ax.set_yticklabels(spks, fontsize=7)
    ax.set_xlabel("evaluated at utterance"); ax.set_ylabel("direction optimised on")
    ax.set_title(r"$\log_{10} R_t$: directions transfer between utterances"
                 "\n" "while being near-orthogonal to each other", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label(f"random direction = {np.log10(rnd):+.2f} (top of scale)", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "fig4_crossexample.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    fig_transfer()
    fig_dimension()
    fig_crossexample()
