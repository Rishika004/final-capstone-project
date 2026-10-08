"""
experiments/make_fig1.py
========================
Figure 1 — the paper's one picture.

PLAN_V3 W9: "Build Fig. 1 first (alpha per encoder vs matched random at
t=50/200) -- it is the paper. All four existing PNGs are old-base and
unusable."

WHAT IT MUST SAY, AND WHAT IT MUST NOT
--------------------------------------
Council v4 A.2 is explicit: **lead C1 with the LEVEL result, not the
ordering.** The level result is that every speaker encoder's own identity
direction survives the purifier WORSE than a random direction matched to it in
both norm and per-mel-band envelope -- on five encoders across four
architecture families, two of which the optimiser never saw.

The ordering (WavLM > ECAPA > {x-vector, d-vector}) is the fragile half: both
middle steps sit at p_adj = .032 at t = 50, the middle collapses entirely at
t = 200, and x-vector vs d-vector never separates (RESULTS 44). So the figure
must NOT look like a ranking chart. It is built as:

  LEFT  the landscape, in absolute alpha: where the encoders sit against their
        own band-matched controls, against isotropic `random`, and against
        `pca_span` (a random direction inside the clean-speech principal
        subspace, which survives BEST -- the purifier preserves the subspace
        clean speech occupies).
  RIGHT the test, as the paired per-speaker difference (encoder - its OWN
        band-matched control) with BCa 95 % CIs. Every interval must exclude
        zero, and that -- not the order of the bars -- is C1.

x-vector is deliberately absent: it exists only on the RESULTS 37 cohort
(`encoder_subspace_matched.csv`) and 43d forbids mixing cohorts in one table.
The d-vector's frozen-VAD row is the 43 front-end control, not a sixth
encoder, so it is excluded here and quoted in the text.

Run:
    python experiments/make_fig1.py
    python experiments/make_fig1.py --no-pca     # drop the pca_span reference
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
SRC = os.path.join(R, "encoder_subspace_13b.csv")

# Order is NOT the message, so the encoders are laid out by architecture
# family rather than by alpha: transformer, TDNN x2, ResNet, GE2E.
ENCODERS = [
    ("WavLM-SV", "WavLM-SV", "transformer", True),
    ("ECAPA-TDNN", "ECAPA-TDNN", "TDNN", True),
    ("CAM++", "CAM++", "TDNN (D-TDNN)", False),
    ("ERes2Net", "ERes2Net", "ResNet", False),
    ("Resemblyzer d-vector", "d-vector", "GE2E LSTM", True),
]
EXCLUDE = {"Resemblyzer d-vector (frozen VAD)",
           "band_matched:Resemblyzer d-vector (frozen VAD)"}


def paired(d, col, enc):
    """per-speaker (encoder - its own band-matched control)"""
    a = d[d.direction == enc].set_index("speaker")[col]
    b = d[d.direction == f"band_matched:{enc}"].set_index("speaker")[col]
    j = pd.concat([a.rename("enc"), b.rename("ctl")], axis=1).dropna()
    return j["enc"].to_numpy(), j["ctl"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--out", default=os.path.join(PROJ, "paper", "fig1_alpha.png"))
    ap.add_argument("--csv", default=os.path.join(R, "fig1_alpha.csv"))
    a = ap.parse_args()

    if not os.path.isfile(SRC):
        sys.exit(f"FATAL: {SRC} missing")
    d = pd.read_csv(SRC)
    d = d[~d.direction.isin(EXCLUDE)]
    nspk = d.speaker.nunique()

    rows = []
    for col, tlab in [("alpha_t50", "t = 50"), ("alpha_t200", "t = 200")]:
        praw, tmp = [], []
        for key, short, fam, seen in ENCODERS:
            e, c = paired(d, col, key)
            dd = e - c
            lo, hi, _, _ = bca_ci(dd, alpha=0.05, n_boot=4000)
            p = wilcoxon_p(dd)
            praw.append(p)
            tmp.append(dict(timestep=tlab, col=col, encoder=short, family=fam,
                            crafted=seen, n=len(dd),
                            alpha=float(e.mean()), control=float(c.mean()),
                            delta=float(dd.mean()), lo=lo, hi=hi, p=p,
                            below=int((dd < 0).sum())))
        for r, pa in zip(tmp, holm(praw)):
            r["p_holm"] = pa
            r["sig"] = bool(pa < 0.05 and r["hi"] < 0)
        rows += tmp
    out = pd.DataFrame(rows)
    out.to_csv(a.csv, index=False)

    refs = {}
    for col in ("alpha_t50", "alpha_t200"):
        refs[col] = {k: float(d[d.direction == k][col].mean())
                     for k in ("random", "pca_span")}

    # ---------------------------------------------------------------- report
    print("=" * 96)
    print(f"FIGURE 1 DATA -- n = {nspk} speakers, purifier v7 DDIM, "
          f"{os.path.basename(SRC)}")
    print("=" * 96)
    for col, tlab in [("alpha_t50", "t = 50"), ("alpha_t200", "t = 200")]:
        g = out[out.col == col]
        print(f"\n  {tlab}     random {refs[col]['random']:.3f}   "
              f"pca_span {refs[col]['pca_span']:.3f}")
        print(f"  {'encoder':12s} {'family':16s} {'crafted':>8s} "
              f"{'alpha':>7s} {'own ctl':>8s} {'delta':>8s} "
              f"{'95% CI':>19s} {'k/n':>7s} {'p_holm':>9s} sig")
        for _, r in g.iterrows():
            print(f"  {r.encoder:12s} {r.family:16s} "
                  f"{'yes' if r.crafted else 'NO':>8s} "
                  f"{r.alpha:7.3f} {r.control:8.3f} {r.delta:+8.3f} "
                  f"[{r.lo:+8.3f},{r.hi:+8.3f}] {int(r.below):3d}/{int(r.n):<3d} "
                  f"{r.p_holm:9.2e} {'*' if r.sig else ''}")
        k = int(g.sig.sum())
        print(f"  -> {k} of {len(g)} encoders below their OWN band-matched "
              f"control (Holm within timestep)")

    k_all = int(out.sig.sum())
    print(f"\n  TOTAL: {k_all} of {len(out)} cells significant "
          f"(5 encoders x 2 timesteps)")
    if k_all == len(out):
        print("  C1's LEVEL claim holds in every cell. That is the figure's")
        print("  message; the ordering is not and must not be read from it.")

    # ------------------------------------------------------------- the plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                   # pragma: no cover
        print(f"\n[fig1] matplotlib unavailable ({e}); CSV written, no PNG")
        return

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200,
    })
    # ONE colour scheme across both panels: colour = TIMESTEP, always.
    # Encoders the optimiser never saw are marked on the tick label with a
    # dagger, never by colour -- using the same two colours for "timestep" in
    # one panel and "crafted against" in the other is unreadable.
    C_T50, C_T200 = "#1b3a6b", "#2e8b8b"
    C_CTL, C_RND, C_PCA = "#9a9a9a", "#c0392b", "#7d6608"
    TS = [("alpha_t50", "t = 50", C_T50, "o", -0.14),
          ("alpha_t200", "t = 200", C_T200, "s", +0.14)]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    xs = np.arange(len(ENCODERS))
    names = [e[1] for e in ENCODERS]
    seen = [e[3] for e in ENCODERS]
    ticks = [n + ("" if s else "$^\\dagger$") for n, s in zip(names, seen)]

    # ---- LEFT: the landscape
    ax = axes[0]
    for col, lab, colr, mk, off in TS:
        g = out[out.col == col].set_index("encoder").loc[names]
        for x, (_, r) in zip(xs, g.iterrows()):
            ax.plot([x + off, x + off], [r.control, r.alpha],
                    color="#aaa", lw=0.9, zorder=1)
        ax.scatter(xs + off, g.control, s=21, marker=mk, facecolor="white",
                   edgecolor=C_CTL, lw=1.0, zorder=3)
        ax.scatter(xs + off, g.alpha, s=23, marker=mk, color=colr,
                   zorder=4, label=lab)
    for col, lab, colr, mk, off in TS:
        ls = "-" if col.endswith("50") else "--"
        ax.axhline(refs[col]["random"], color=C_RND, lw=0.8, ls=ls, zorder=0)
        if not a.no_pca:
            ax.axhline(refs[col]["pca_span"], color=C_PCA, lw=0.8, ls=ls,
                       zorder=0)
    ax.text(-0.42, refs["alpha_t50"]["random"] + 0.014, "isotropic random",
            color=C_RND, fontsize=7.2, ha="left")
    if not a.no_pca:
        ax.text(-0.42, refs["alpha_t50"]["pca_span"] + 0.014,
                "clean-speech PCA span", color=C_PCA, fontsize=7.2, ha="left")
    ax.scatter([], [], s=21, marker="o", facecolor="white",
               edgecolor=C_CTL, lw=1.0, label="band-matched control")
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(ENCODERS) - 0.4)
    ax.set_ylabel(r"survival coefficient $\alpha$")
    ax.set_title("(a) every encoder sits below its own\n"
                 "band-matched random control", loc="left")
    ax.set_ylim(0.20, 0.95)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.25)

    # ---- RIGHT: the test
    ax = axes[1]
    for col, lab, colr, mk, off in TS:
        g = out[out.col == col].set_index("encoder").loc[names]
        ax.errorbar(g.delta, xs + off,
                    xerr=[g.delta - g.lo, g.hi - g.delta],
                    fmt=mk, ms=4.2, lw=1.0, capsize=2.2, color=colr, label=lab)
    ax.axvline(0.0, color="k", lw=0.9)
    ax.set_yticks(xs)
    ax.set_yticklabels(ticks)
    ax.set_ylim(len(ENCODERS) - 0.45, -0.55)        # inverted, with headroom
    ax.set_xlim(-0.40, 0.055)
    ax.set_xlabel(r"$\alpha_{\mathrm{encoder}}-\alpha_{\mathrm{band\,matched}}$"
                  "  (paired, per speaker)")
    ax.set_title("(b) every interval excludes zero\n"
                 f"BCa 95 % CI, n = {nspk}, Holm within timestep", loc="left")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.25)

    fig.text(0.012, -0.055,
             "$^\\dagger$ never in the crafting ensemble: these two encoders "
             "were measured, not optimised against.",
             fontsize=7.2, color="#333")

    fig.tight_layout(pad=0.6)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"\nwritten: {a.out}\nwritten: {a.csv}")


if __name__ == "__main__":
    main()
