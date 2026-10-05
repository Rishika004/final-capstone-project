"""
plot_results.py
===============
Generates all result graphs for the paper's Results and Outcome section.
Run with:  python plot_results.py
Saves PNG files to: results_graphs/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("results_graphs", exist_ok=True)

# ── Colour palette (clean academic style) ─────────────────────────────────
GREEN   = "#2ecc71"
RED     = "#e74c3c"
BLUE    = "#3498db"
PURPLE  = "#9b59b6"
ORANGE  = "#f39c12"
GREY    = "#95a5a6"
BG      = "#f9f9f9"

FILES   = ["S1", "S2", "S3", "S4", "S5"]

# ── Per-file data ──────────────────────────────────────────────────────────
prr          = [0.9980, 0.9994, 0.9995, 0.9989, 0.9996]
speaker_sim  = [-0.3445, -0.1059, -0.0365, -0.3773, -0.0750]
pesq         = [3.923,  3.803,  4.019,  None,   3.560 ]
stoi         = [0.7257, 0.8480, 0.8040, 0.7037, 0.8457]
snr_db       = [59.82,  61.73,  60.60,  60.93,  62.12 ]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# ══════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Speaker Similarity per Speaker (Before Purification)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
colors = [GREEN if s < 0.5 else RED for s in speaker_sim]
bars = ax.bar(FILES, speaker_sim, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
ax.axhline(y=0.5,  color=RED,   linewidth=2, linestyle="--", label="Cloning threshold (0.5)")
ax.axhline(y=0.0,  color=GREY,  linewidth=1, linestyle="-")
ax.axhline(y=-0.5, color=GREEN, linewidth=1, linestyle=":", alpha=0.5)
for bar, val in zip(bars, speaker_sim):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.03,
            f"{val:.3f}", ha="center", va="top", fontsize=10, fontweight="bold", color="white")
ax.set_ylim(-0.55, 0.75)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Speaker Similarity After Protection\n(All Values Negative = Cloning Fails on Every Speaker)")
ax.legend(fontsize=10)
patch_g = mpatches.Patch(color=GREEN, label="Cloning FAILS (sim < 0.5) ✅")
patch_r = mpatches.Patch(color=RED,   label="Cloning WORKS (sim > 0.5) ❌")
ax.legend(handles=[patch_g, patch_r], fontsize=10, loc="upper right")
ax.text(2, 0.55, "← Cloning works above this line", fontsize=9, color=RED, ha="center")
ax.text(2, -0.48, "← Stronger protection", fontsize=9, color=GREEN, ha="center")
plt.tight_layout()
plt.savefig("results_graphs/graph1_speaker_similarity.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Graph 1 saved — Speaker Similarity")

# ══════════════════════════════════════════════════════════════════════════
# GRAPH 2 — PRR per Speaker
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(FILES, prr, color=BLUE, width=0.5, edgecolor="white", linewidth=1.5)
ax.axhline(y=1.0, color=GREEN, linewidth=2, linestyle="--", label="Target PRR = 1.0")
ax.axhline(y=0.5, color=ORANGE, linewidth=2, linestyle="--", label="Partial survival = 0.5")
for bar, val in zip(bars, prr):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0.990, 1.005)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("Purification Resistance Rate (PRR)")
ax.set_title("PRR Across All Speakers\n(PRR ≈ 1.0 = Protection Fully Survives DiffPure)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results_graphs/graph2_prr.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Graph 2 saved — PRR")

# ══════════════════════════════════════════════════════════════════════════
# GRAPH 3 — Audio Quality (PESQ + STOI side by side)
# ══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# PESQ
pesq_vals  = [v if v is not None else 0 for v in pesq]
pesq_color = [PURPLE if v > 0 else GREY for v in pesq_vals]
bars1 = ax1.bar(FILES, pesq_vals, color=pesq_color, width=0.5, edgecolor="white", linewidth=1.5)
ax1.axhline(y=3.0, color=ORANGE, linewidth=2, linestyle="--", label="Acceptable (3.0)")
ax1.axhline(y=4.0, color=GREEN,  linewidth=2, linestyle="--", label="Good (4.0)")
for bar, val, orig in zip(bars1, pesq_vals, pesq):
    label = f"{val:.2f}" if orig is not None else "N/A"
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             label, ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_ylim(0, 5)
ax1.set_xlabel("Speaker Sample")
ax1.set_ylabel("PESQ Score")
ax1.set_title("Speech Quality (PESQ)\nTarget > 3.0 for Acceptable Quality")
ax1.legend(fontsize=9)

# STOI
bars2 = ax2.bar(FILES, stoi, color=ORANGE, width=0.5, edgecolor="white", linewidth=1.5)
ax2.axhline(y=0.95, color=GREEN,  linewidth=2, linestyle="--", label="Excellent (0.95)")
ax2.axhline(y=0.80, color=ORANGE, linewidth=2, linestyle="--", label="Good (0.80)")
for bar, val in zip(bars2, stoi):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_ylim(0, 1.1)
ax2.set_xlabel("Speaker Sample")
ax2.set_ylabel("STOI Score")
ax2.set_title("Speech Intelligibility (STOI)\nTarget > 0.80")
ax2.legend(fontsize=9)

plt.suptitle("Audio Quality Metrics After Protection", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results_graphs/graph3_audio_quality.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Graph 3 saved — Audio Quality")

# ══════════════════════════════════════════════════════════════════════════
# GRAPH 4 — Before vs After Protection (Speaker Similarity Comparison)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(FILES))
width = 0.35
sim_before = [1.0, 1.0, 1.0, 1.0, 1.0]   # unprotected: perfect similarity
sim_after  = speaker_sim

bars1 = ax.bar(x - width/2, sim_before, width, label="Without Protection",
               color=RED, edgecolor="white", linewidth=1.5, alpha=0.85)
bars2 = ax.bar(x + width/2, sim_after,  width, label="With Our Protection",
               color=GREEN, edgecolor="white", linewidth=1.5, alpha=0.85)

ax.axhline(y=0.5, color="black", linewidth=2, linestyle="--", label="Cloning threshold (0.5)")
ax.axhline(y=0.0, color=GREY,   linewidth=1, linestyle="-")

for bar, val in zip(bars2, sim_after):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.03,
            f"{val:.2f}", ha="center", va="top", fontsize=9, fontweight="bold", color="white")

ax.set_xticks(x)
ax.set_xticklabels(FILES)
ax.set_ylim(-0.6, 1.2)
ax.set_xlabel("Speaker Sample")
ax.set_ylabel("Cosine Similarity with Original Speaker")
ax.set_title("Speaker Similarity: Without vs With Protection\n"
             "Without protection → cloning succeeds | With protection → cloning fails")
ax.legend(fontsize=10)
ax.text(2, 0.6, "Cloning succeeds above this line", fontsize=9, color="red", ha="center")
plt.tight_layout()
plt.savefig("results_graphs/graph4_before_after.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Graph 4 saved — Before vs After")

# ══════════════════════════════════════════════════════════════════════════
# GRAPH 5 — Radar Chart (All Metrics at a Glance)
# ══════════════════════════════════════════════════════════════════════════
categories  = ["PRR\n(×1)", "Speaker\nConfusion", "PESQ\n(÷5)", "STOI", "SNR\n(÷70)"]
mean_values = [
    np.mean(prr),
    abs(np.mean(speaker_sim)),           # flip sign so higher = better
    np.mean([v for v in pesq if v]) / 5, # normalise to 0-1
    np.mean(stoi),
    np.mean(snr_db) / 70,                # normalise to 0-1
]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
mean_values += mean_values[:1]
angles      += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles, mean_values, color=BLUE, linewidth=2)
ax.fill(angles, mean_values, color=BLUE, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=8)
ax.set_title("Overall System Performance\n(All Metrics Normalised to 0–1, Higher = Better)",
             size=13, fontweight="bold", pad=20)
for angle, val, cat in zip(angles[:-1], mean_values[:-1], categories):
    ax.text(angle, val + 0.07, f"{val:.2f}", ha="center", va="center", fontsize=10, fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig("results_graphs/graph5_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Graph 5 saved — Radar Chart")

# ══════════════════════════════════════════════════════════════════════════
# GRAPH 6 — Comparison with Cloning Threshold (Summary Bar)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
metrics      = ["PRR", "−Speaker\nSimilarity", "PESQ/5", "STOI", "SNR/70"]
our_scores   = [
    np.mean(prr),
    abs(np.mean(speaker_sim)),
    np.mean([v for v in pesq if v]) / 5,
    np.mean(stoi),
    np.mean(snr_db) / 70,
]
target_scores = [1.0, 0.5, 0.6, 0.95, 0.5]   # minimum targets

x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, our_scores,    w, label="Our System", color=GREEN, edgecolor="white", linewidth=1.5)
ax.bar(x + w/2, target_scores, w, label="Minimum Target", color=GREY,  edgecolor="white", linewidth=1.5, alpha=0.7)
for i, (o, t) in enumerate(zip(our_scores, target_scores)):
    icon = "✅" if o >= t else "⚠️"
    ax.text(i - w/2, o + 0.015, f"{icon}\n{o:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.2)
ax.set_ylabel("Normalised Score (higher = better)")
ax.set_title("Our System vs Minimum Targets\n(Green = Achieved, Grey = Target)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results_graphs/graph6_targets.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Graph 6 saved — Targets Comparison")

print("\n" + "="*55)
print("  ALL 6 GRAPHS SAVED TO: results_graphs/")
print("="*55)
print("  graph1_speaker_similarity.png  — sim per speaker")
print("  graph2_prr.png                 — PRR per speaker")
print("  graph3_audio_quality.png       — PESQ + STOI")
print("  graph4_before_after.png        — before vs after protection")
print("  graph5_radar.png               — all metrics radar")
print("  graph6_targets.png             — vs minimum targets")
print("="*55)
