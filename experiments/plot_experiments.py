"""
experiments/plot_experiments.py
===============================
Generate publication-style graphs from the experiment CSVs:
  - robustness.csv   -> robustness_plot.png   (cos-sim per transform)
  - comparison.csv   -> comparison_plot.png    (range-energy + cos per method)
  - ablation.csv     -> ablation_plot.png      (4-panel sweep)
  - transfer.csv     -> transfer_plot.png      (cross-encoder cos-sim)

Run: python experiments/plot_experiments.py
Output: experiments/results/*.png
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.csvio import read_csv as read_csv_safe

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

GREEN, RED, BLUE, ORANGE, PURPLE, GREY = (
    "#27ae60", "#e74c3c", "#2980b9", "#e67e22", "#8e44ad", "#95a5a6")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.titlesize": 12, "axes.labelsize": 10})


def plot_robustness():
    f = os.path.join(R, "robustness.csv")
    if not os.path.exists(f): return
    df = read_csv_safe(f)
    g = df.groupby("transform", sort=False)["cos_sim_after"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GREEN if v < 0.5 else RED for v in g.values]
    ax.bar(range(len(g)), g.values, color=colors, edgecolor="white", linewidth=1.5)
    ax.axhline(0.5, color=RED, ls="--", lw=2, label="cloning threshold (0.5)")
    ax.axhline(0.0, color=GREY, lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("mean cosine similarity (WavLM)")
    ax.set_title("Robustness to real-world transforms\n"
                 "green = protection holds (cos < 0.5), red = stripped")
    for i, v in enumerate(g.values):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(R, "robustness_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved robustness_plot.png")


def plot_transfer():
    f = os.path.join(R, "transfer.csv")
    if not os.path.exists(f): return
    df = read_csv_safe(f)
    g = df.groupby("encoder", sort=False).agg(cos=("cos_sim", "mean"),
                                              thr=("threshold", "first"))
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(g))
    colors = [GREEN if r.cos < r.thr else RED for _, r in g.iterrows()]
    ax.bar(x, g["cos"].values, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    # per-encoder threshold markers
    for i, (_, r) in enumerate(g.iterrows()):
        ax.plot([i - 0.3, i + 0.3], [r.thr, r.thr], color="black", ls="--", lw=1.8)
        ax.text(i, r.thr + 0.03, f"thr={r.thr}", ha="center", fontsize=8)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(g.index, fontsize=10)
    ax.set_ylabel("cos( E(original), E(protected) )")
    ax.set_ylim(-0.6, 1.05)
    ax.set_title("Cross-encoder transferability\n"
                 "delta crafted vs WavLM only  |  green = fools encoder, red = does NOT transfer")
    for i, v in enumerate(g["cos"].values):
        ax.text(i, v + (0.03 if v >= 0 else -0.05), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(R, "transfer_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved transfer_plot.png")


def plot_comparison():
    f = os.path.join(R, "comparison.csv")
    if not os.path.exists(f): return
    df = read_csv_safe(f)
    g = df.groupby("method", sort=False).agg(
        cos=("speaker_cos_sim", "mean"), rf=("range_energy_frac", "mean"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    labels = [m.split(". ", 1)[-1] for m in g.index]
    a1.barh(range(len(g)), g["rf"].values * 100, color=BLUE, edgecolor="white")
    a1.set_yticks(range(len(g))); a1.set_yticklabels(labels, fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel("range-space energy (%)  [DiffPure erases this]")
    a1.set_title("Purification exposure\nlower = more survives DiffPure")
    for i, v in enumerate(g["rf"].values * 100):
        a1.text(v + 0.01, i, f"{v:.2f}%", va="center", fontsize=9)

    cosv = g["cos"].values
    a2.barh(range(len(g)), np.nan_to_num(cosv), color=[GREEN if (not np.isnan(c) and c < 0.5) else GREY for c in cosv],
            edgecolor="white")
    a2.axvline(0.5, color=RED, ls="--", lw=2)
    a2.set_yticks(range(len(g))); a2.set_yticklabels(labels, fontsize=9)
    a2.invert_yaxis()
    a2.set_xlabel("speaker cos-sim (WavLM)  [< 0.5 fools encoder]")
    a2.set_title("Speaker confusion\nlower = fools WavLM")
    for i, c in enumerate(cosv):
        a2.text((c if not np.isnan(c) else 0) + 0.02, i,
                "n/a" if np.isnan(c) else f"{c:+.2f}", va="center", fontsize=9)
    plt.suptitle("2x2 ablation: null-projection zeroes purification exposure; "
                 "PGD drives speaker confusion", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(R, "comparison_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved comparison_plot.png")


def plot_ablation():
    f = os.path.join(R, "ablation.csv")
    if not os.path.exists(f): return
    df = read_csv_safe(f)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [("K", "K", "null_dim", axes[0, 0], "null-space dimension vs K"),
             ("pgd_steps", "pgd_steps", "cos_sim", axes[0, 1], "speaker confusion vs PGD steps"),
             ("epsilon", "epsilon", "snr_db", axes[1, 0], "SNR (imperceptibility) vs epsilon"),
             ("threshold", "threshold", "prr", axes[1, 1], "PRR vs SVD threshold")]
    for sweep, param, metric, ax, title in specs:
        sub = df[df.sweep == sweep].groupby(param)[metric].mean()
        ax.plot(range(len(sub)), sub.values, "o-", color=BLUE, lw=2, markersize=8)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([str(x) for x in sub.index])
        ax.set_xlabel(param); ax.set_ylabel(metric); ax.set_title(title)
        for i, v in enumerate(sub.values):
            ax.text(i, v, f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
    plt.suptitle("Hyperparameter ablation (sample_001)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(R, "ablation_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved ablation_plot.png")


def plot_transfer_compare():
    """Before (single-encoder) vs after (ensemble PGD) per encoder."""
    f1 = os.path.join(R, "transfer.csv")
    f2 = os.path.join(R, "transfer_multi.csv")
    if not (os.path.exists(f1) and os.path.exists(f2)): return
    d1 = read_csv_safe(f1).groupby("encoder", sort=False).agg(
        cos=("cos_sim", "mean"), thr=("threshold", "first"))
    d2 = read_csv_safe(f2).groupby("encoder", sort=False).agg(cos=("cos_sim", "mean"))
    order = ["WavLM (target)", "ECAPA-TDNN", "Resemblyzer d-vector"]
    d1 = d1.reindex(order); d2 = d2.reindex(order)
    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w/2, d1["cos"].values, w, label="single-encoder PGD (vs WavLM only)",
                color=GREY, edgecolor="white")
    b2 = ax.bar(x + w/2, d2["cos"].values, w, label="ensemble PGD (ours)",
                color=GREEN, edgecolor="white")
    for i, r in enumerate(d1.itertuples()):
        ax.plot([i - w/2 - 0.18, i + w/2 + 0.18], [r.thr, r.thr], color=RED, ls="--", lw=1.6)
        ax.text(i + w/2 + 0.20, r.thr, f"thr={r.thr}", va="center", fontsize=8, color=RED)
    ax.axhline(0, color="black", lw=0.8)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + (0.02 if v >= 0 else -0.05),
                    f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("cos( E(original), E(protected) )  [real eval]")
    ax.set_ylim(-0.95, 1.05)
    ax.set_title("Cross-encoder transferability: single-encoder vs ensemble PGD\n"
                 "below the red threshold = cloning fails on that encoder")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(R, "transfer_compare_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved transfer_compare_plot.png")


if __name__ == "__main__":
    plot_robustness()
    plot_comparison()
    plot_ablation()
    plot_transfer()
    plot_transfer_compare()
    print("done")
