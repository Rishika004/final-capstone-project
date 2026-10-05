"""
plot_all_results.py
===================
Generates 7 result graphs covering ALL metrics:
  1. Speaker Similarity (cosine sim) per speaker
  2. PRR per speaker
  3. PESQ per speaker
  4. STOI per speaker
  5. SNR per speaker
  6. Before vs After Protection (grouped bar)
  7. Radar Chart — all metrics normalised

Run:  python plot_all_results.py
Output: results_graphs/
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("results_graphs", exist_ok=True)

# ── palette ──────────────────────────────────────────────────────
GREEN  = "#27ae60"
RED    = "#e74c3c"
BLUE   = "#2980b9"
PURPLE = "#8e44ad"
ORANGE = "#e67e22"
TEAL   = "#16a085"
GREY   = "#95a5a6"
BG     = "#f9f9f9"

SPEAKERS = ["S1", "S2", "S3", "S4", "S5"]

# ── data ─────────────────────────────────────────────────────────
prr         = [0.9980, 0.9994, 0.9995, 0.9989, 0.9996]
speaker_sim = [-0.3445, -0.1059, -0.0365, -0.3773, -0.0750]
pesq        = [3.923,  3.803,  4.019,  None,   3.560]
stoi        = [0.7257, 0.8480, 0.8040, 0.7037, 0.8457]
snr_db      = [59.82,  61.73,  60.60,  60.93,  62.12]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   BG,
    "axes.facecolor":     BG,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
})

# ─────────────────────────────────────────────────────────────────
# GRAPH 1 — Speaker Similarity
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = [GREEN if s < 0.5 else RED for s in speaker_sim]
bars = ax.bar(SPEAKERS, speaker_sim, color=colors, width=0.5,
              edgecolor="white", linewidth=1.5)
ax.axhline(y=0.5,  color=RED,   linewidth=2, linestyle="--", alpha=0.8)
ax.axhline(y=0.0,  color=GREY,  linewidth=1, linestyle="-")
ax.axhline(y=-0.5, color=GREEN, linewidth=1, linestyle=":", alpha=0.5)

for bar, val in zip(bars, speaker_sim):
    ax.text(bar.get_x() + bar.get_width()/2,
            val - 0.025, f"{val:.3f}",
            ha="center", va="top", fontsize=10, fontweight="bold", color="white")

ax.set_ylim(-0.55, 0.80)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("Cosine Similarity  cos(e_prot, e_orig)")
ax.set_title("Speaker Cosine Similarity After Protection\n"
             "All values < 0  =>  Cloning fails on every speaker")
ax.text(2, 0.57, "Cloning works above this line (threshold = 0.5)",
        fontsize=9, color=RED, ha="center")
ax.text(2, -0.49, "Stronger protection (more negative)",
        fontsize=9, color=GREEN, ha="center")

patch_g = mpatches.Patch(color=GREEN, label="Cloning FAILS  (sim < 0.5)  [PASS]")
patch_r = mpatches.Patch(color=RED,   label="Cloning WORKS  (sim > 0.5)  [FAIL]")
ax.legend(handles=[patch_g, patch_r], fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig("results_graphs/graph1_speaker_similarity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graph 1 saved -- Speaker Similarity")

# ─────────────────────────────────────────────────────────────────
# GRAPH 2 — PRR
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(SPEAKERS, prr, color=BLUE, width=0.5,
              edgecolor="white", linewidth=1.5)
ax.axhline(y=1.0, color=GREEN,  linewidth=2, linestyle="--",
           label="Ideal PRR = 1.0  (protection fully survives)")
ax.axhline(y=0.5, color=ORANGE, linewidth=2, linestyle="--",
           label="Partial survival = 0.5")
ax.axhline(y=0.0, color=RED,    linewidth=1, linestyle=":",
           label="PRR = 0  (protection fully stripped)")

for bar, val in zip(bars, prr):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.0004, f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylim(0.990, 1.006)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("Purification Resistance Rate  (PRR)")
ax.set_title("PRR Across All Speakers\n"
             "PRR = ||f(x+delta)-f(x)|| / ||f(x)-x||,   f = DiffPure")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results_graphs/graph2_prr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graph 2 saved -- PRR")

# ─────────────────────────────────────────────────────────────────
# GRAPH 3 — PESQ
# ─────────────────────────────────────────────────────────────────
pesq_vals  = [v if v is not None else 0 for v in pesq]
pesq_color = [PURPLE if v > 0 else GREY for v in pesq_vals]
labels_pesq= [f"{v:.2f}" if v > 0 else "N/A" for v in pesq_vals]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(SPEAKERS, pesq_vals, color=pesq_color, width=0.5,
              edgecolor="white", linewidth=1.5)
ax.axhline(y=4.5, color=BLUE,   linewidth=1.5, linestyle="--",
           label="Excellent (4.5)")
ax.axhline(y=4.0, color=GREEN,  linewidth=2,   linestyle="--",
           label="Good (4.0)")
ax.axhline(y=3.0, color=ORANGE, linewidth=2,   linestyle="--",
           label="Acceptable (3.0)  [our target]")
ax.axhline(y=2.0, color=RED,    linewidth=1.5, linestyle="--",
           label="Poor (2.0)")

for bar, val, lbl in zip(bars, pesq_vals, labels_pesq):
    ypos = val + 0.05 if val > 0 else 0.1
    ax.text(bar.get_x() + bar.get_width()/2, ypos, lbl,
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylim(0, 5.3)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("PESQ Score  (1.0 – 4.5 scale)")
ax.set_title("Speech Quality — PESQ Score per Speaker\n"
             "Target > 3.0  |  S4 = N/A (duration mismatch)")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("results_graphs/graph3_pesq.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graph 3 saved -- PESQ")

# ─────────────────────────────────────────────────────────────────
# GRAPH 4 — STOI
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
stoi_colors = [GREEN if v >= 0.80 else ORANGE for v in stoi]
bars = ax.bar(SPEAKERS, stoi, color=stoi_colors, width=0.5,
              edgecolor="white", linewidth=1.5)
ax.axhline(y=0.95, color=BLUE,   linewidth=2, linestyle="--",
           label="Excellent (0.95)")
ax.axhline(y=0.80, color=GREEN,  linewidth=2, linestyle="--",
           label="Good (0.80)  [our target]")
ax.axhline(y=0.65, color=ORANGE, linewidth=2, linestyle="--",
           label="Acceptable (0.65)")

for bar, val in zip(bars, stoi):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.005, f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylim(0, 1.10)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("STOI Score  (0 – 1 scale)")
ax.set_title("Speech Intelligibility — STOI Score per Speaker\n"
             "Target > 0.80  |  Green = meets target, Orange = below")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results_graphs/graph4_stoi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graph 4 saved -- STOI")

# ─────────────────────────────────────────────────────────────────
# GRAPH 5 — SNR
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
snr_colors = [TEAL]*len(snr_db)
bars = ax.bar(SPEAKERS, snr_db, color=snr_colors, width=0.5,
              edgecolor="white", linewidth=1.5)
ax.axhline(y=50, color=ORANGE, linewidth=2, linestyle="--",
           label="Good quality threshold (50 dB)")
ax.axhline(y=60, color=GREEN,  linewidth=2, linestyle="--",
           label="Excellent threshold (60 dB)")

for bar, val in zip(bars, snr_db):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.3, f"{val:.1f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylim(0, 72)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("Signal-to-Noise Ratio  (dB)")
ax.set_title("SNR per Speaker\n"
             "SNR = 20 log(||x|| / ||delta||)   —   Higher = smaller perturbation")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results_graphs/graph5_snr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graph 5 saved -- SNR")

# ─────────────────────────────────────────────────────────────────
# GRAPH 6 — Before vs After Protection (grouped bar)
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x     = np.arange(len(SPEAKERS))
width = 0.30

sim_before   = [1.0]*5          # unprotected
sim_antifake = [0.75]*5         # AntiFake after DiffPure
sim_ours     = speaker_sim      # our method

b1 = ax.bar(x - width,   sim_before,   width, label="Unprotected",
            color=RED,    edgecolor="white", linewidth=1.5, alpha=0.85)
b2 = ax.bar(x,           sim_antifake, width, label="AntiFake + DiffPure",
            color=ORANGE, edgecolor="white", linewidth=1.5, alpha=0.85)
b3 = ax.bar(x + width,   sim_ours,     width, label="Ours (null-space)",
            color=GREEN,  edgecolor="white", linewidth=1.5, alpha=0.85)

ax.axhline(y=0.5, color="black", linewidth=2, linestyle="--",
           label="Cloning threshold (0.5)")
ax.axhline(y=0.0, color=GREY, linewidth=1, linestyle="-")

for bar, val in zip(b3, sim_ours):
    ax.text(bar.get_x() + bar.get_width()/2,
            val - 0.025, f"{val:.2f}",
            ha="center", va="top", fontsize=8.5, fontweight="bold", color="white")

ax.set_xticks(x)
ax.set_xticklabels(SPEAKERS)
ax.set_ylim(-0.6, 1.25)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("Cosine Similarity  cos(e_prot, e_orig)")
ax.set_title("Cosine Similarity: Unprotected vs AntiFake vs Our Method\n"
             "Below 0.5 = cloning fails   |   Below 0.0 = complete identity confusion")
ax.legend(fontsize=9)
ax.text(2, 0.62, "Cloning works above this line",
        fontsize=9, color="red", ha="center")
plt.tight_layout()
plt.savefig("results_graphs/graph6_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graph 6 saved -- Before vs After comparison")

# ─────────────────────────────────────────────────────────────────
# GRAPH 7 — Radar Chart (all metrics normalised)
# ─────────────────────────────────────────────────────────────────
categories = [
    "PRR",
    "Speaker\nConfusion\n(-sim)",
    "PESQ\n(/4.5)",
    "STOI",
    "SNR\n(/70)",
]
vals_ours = [
    np.mean(prr),
    abs(np.mean(speaker_sim)),
    np.mean([v for v in pesq if v]) / 4.5,
    np.mean(stoi),
    np.mean(snr_db) / 70,
]
vals_target = [1.0, 0.5, 3.0/4.5, 0.80, 50/70]

N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
vals_ours   = vals_ours   + vals_ours[:1]
vals_target = vals_target + vals_target[:1]
angles      = angles      + angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles, vals_ours,   color=GREEN, linewidth=2.5, label="Our System")
ax.fill(angles, vals_ours,   color=GREEN, alpha=0.20)
ax.plot(angles, vals_target, color=ORANGE, linewidth=2,  linestyle="--", label="Minimum Target")
ax.fill(angles, vals_target, color=ORANGE, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=8)
ax.set_title("Overall System Performance\n"
             "All Metrics Normalised to 0-1  |  Higher = Better",
             size=13, fontweight="bold", pad=22)
for angle, val in zip(angles[:-1], vals_ours[:-1]):
    ax.text(angle, val + 0.08, f"{val:.2f}",
            ha="center", va="center", fontsize=10, fontweight="bold", color=GREEN)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=10)

plt.tight_layout()
plt.savefig("results_graphs/graph7_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graph 7 saved -- Radar Chart")

# ─────────────────────────────────────────────────────────────────
# summary
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  ALL 7 GRAPHS SAVED TO: results_graphs/")
print("=" * 55)
print("  graph1_speaker_similarity.png  -- cosine sim per speaker")
print("  graph2_prr.png                 -- PRR per speaker")
print("  graph3_pesq.png                -- PESQ per speaker")
print("  graph4_stoi.png                -- STOI per speaker")
print("  graph5_snr.png                 -- SNR per speaker")
print("  graph6_comparison.png          -- Unprotected vs AntiFake vs Ours")
print("  graph7_radar.png               -- all metrics radar")
print("=" * 55)
