"""
experiments/make_fig_cloners.py
===============================
Figure 2 of the defence paper: RESULTS 61c in one picture.

The same two protected arms, the same 29 speakers, the same budget (16.206 vs
16.113 dB SNR, paired p = 0.408, with the purifier-aware arm at BETTER PESQ) --
and the verdict on purifier-in-the-loop crafting flips with the cloner:

  DiffVC   conditions on the Resemblyzer d-vector, which BOTH arms crafted
           against. 2 of 9 cells favour purifier-awareness; the two
           significant cells go AGAINST it, on the d-vector itself.
  F5-TTS   has no speaker encoder and was targeted by neither arm.
           8 of 9 favour purifier-awareness, 5 significantly.

Plotted as the paired delta (purifier-aware minus plain) with its BCa interval,
so the reader sees the effect size and the uncertainty rather than two bars
whose difference they have to infer. Zero is drawn, and the +/- noise floor
(0.0247, RESULTS 34) is shaded, because several cells sit inside it.

Cells where both arms clone under 10 % are drawn hollow: the binary outcome is
saturated there and the continuous margin is all that is left (RESULTS 59b,
61b).

Run:
  python experiments/make_fig_cloners.py
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
from scipy.stats import ttest_rel

from experiments.analyse_matched_control import bca_ci, holm

R = os.path.join(PROJ, "experiments", "results")
OUT = os.path.join(PROJ, "paper", "fig_cloners.png")
SHIP, PLAIN = "fullnsT_tc_m30_seg", "plain_tc_m30_seg"
ENCS = [("ECAPA-TDNN", "ECAPA"), ("WavLM-SV", "WavLM-SV"),
        ("Resemblyzer d-vector", "d-vector")]
PURS = ["none", "ours", "diffwave"]
PURLBL = {"none": "no purifier", "ours": "DiffPure v7", "diffwave": "DiffWave"}
NOISE_FLOOR = 0.0247
FLOOR_PCT = 10.0
SOURCES = [("DiffVC", "cloning_tc_m30_seg_n5.csv",
            "conditions on the d-vector\n(both arms crafted against it)"),
           ("F5-TTS", "cloning_f5tts.csv",
            "no speaker encoder\n(targeted by neither arm)")]


def cells(path):
    d = pd.read_csv(os.path.join(R, path))
    out = []
    for pur in PURS:
        for enc, _ in ENCS:
            s = d[(d.encoder == enc) & (d.purifier == pur)]
            a = s[s.variant == PLAIN].set_index("speaker").margin
            b = s[s.variant == SHIP].set_index("speaker").margin
            j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
            if len(j) < 6:
                continue
            diff = (j.b - j.a).to_numpy()
            _, p = ttest_rel(j.b, j.a)
            lo, hi = bca_ci(diff)[:2]
            ra = 100 * s[s.variant == PLAIN].clone_succeeds.mean()
            rb = 100 * s[s.variant == SHIP].clone_succeeds.mean()
            out.append(dict(purifier=pur, encoder=enc, delta=float(diff.mean()),
                            lo=float(lo), hi=float(hi), p=float(p),
                            floored=bool(ra < FLOOR_PCT and rb < FLOOR_PCT)))
    for r, pa in zip(out, holm([x["p"] for x in out])):
        r["p_holm"] = float(pa)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:                                   # pragma: no cover
        print(f"[fig] matplotlib unavailable ({e})")
        return

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200,
    })
    # Colour = ENCODER, identical in both panels. The panels differ by cloner,
    # which is the whole point and is carried by the titles.
    CE = {"ECAPA-TDNN": "#1b3a6b", "WavLM-SV": "#2e8b8b",
          "Resemblyzer d-vector": "#c0623a"}
    OFF = {"ECAPA-TDNN": -0.22, "WavLM-SV": 0.0, "Resemblyzer d-vector": +0.22}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    rows = []
    for ax, (lbl, path, sub) in zip(axes, SOURCES):
        res = cells(path)
        rows += [dict(cloner=lbl, **r) for r in res]
        n_neg = sum(1 for r in res if r["delta"] < 0)
        ax.axhspan(-NOISE_FLOOR, NOISE_FLOOR, color="#000000", alpha=0.05,
                   zorder=0, linewidth=0)
        ax.axhline(0, color="#444", linewidth=0.8, zorder=1)
        for i, pur in enumerate(PURS):
            for enc, _ in ENCS:
                r = next((x for x in res if x["purifier"] == pur
                          and x["encoder"] == enc), None)
                if r is None:
                    continue
                x = i + OFF[enc]
                ax.errorbar(x, r["delta"],
                            yerr=[[r["delta"] - r["lo"]], [r["hi"] - r["delta"]]],
                            fmt="o", ms=5.5, color=CE[enc], capsize=2.5,
                            linewidth=1.2, zorder=3,
                            markerfacecolor=("white" if r["floored"]
                                             else CE[enc]))
                if r["p_holm"] < .05:
                    ax.text(x, r["hi"] + 0.004, "*", ha="center", fontsize=11,
                            color=CE[enc])
        ax.set_xticks(range(len(PURS)))
        ax.set_xticklabels([PURLBL[p] for p in PURS])
        ax.set_title(f"{lbl}\n{sub}", fontsize=9)
        # Put the count in whichever half the data has vacated, so it never
        # lands on an error bar.
        up = np.mean([r["delta"] for r in res]) < 0
        ax.text(0.97, 0.97 if up else 0.03,
                f"{n_neg}/9 favour\npurifier-in-loop",
                transform=ax.transAxes, ha="right",
                va="top" if up else "bottom", fontsize=7.5, color="#333")

    axes[0].set_ylabel("margin: purifier-in-loop − plain\n(negative = it helps)")
    h = [Line2D([], [], marker="o", ls="", color=CE[e], label=l)
         for e, l in ENCS]
    h += [Line2D([], [], marker="o", ls="", color="#666",
                 markerfacecolor="white", label="both arms < 10 % (saturated)"),
          Line2D([], [], color="#000000", alpha=0.18, linewidth=7,
                 label=f"±{NOISE_FLOOR} noise floor")]
    # Figure-level legend below the axes: inside either panel it sits on data.
    fig.legend(handles=h, frameon=False, fontsize=7.6, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 0.045))

    fig.suptitle("Same arms, same budget — the verdict flips with the cloner",
                 fontsize=10, y=1.01)
    fig.text(0.5, -0.135,
             "29 cross-session test-clean speakers; arms matched at 16.206 vs "
             "16.113 dB SNR (p = 0.408), purifier-aware at better PESQ. "
             "* = Holm p < 0.05 within each panel's nine cells.",
             ha="center", fontsize=7, color="#333", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out, bbox_inches="tight")
    csv = args.out.rsplit(".", 1)[0] + ".csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"[fig] {args.out}\n[fig] {csv}")
    t = pd.DataFrame(rows).pivot_table(index=["purifier", "encoder"],
                                       columns="cloner", values="delta")
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
