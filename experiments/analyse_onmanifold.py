"""
experiments/analyse_onmanifold.py
=================================
Turn the on-manifold probe into the paper's central figure and tables.

Combines two families measured under the SAME attacker (purifier v3 + DDIM,
20 held-out speakers, channel-calibrated EER thresholds):

  additive        plain / EOT / null-regularised crafting   (eval_psr_v3.csv)
  on-manifold     tilt / eq / warp / reverb sweeps          (parametric_probe.csv)

and plots protection against survival for both. If the on-manifold family lands
at high survival and low protection, the anti-correlation we measured within the
additive family extends to the case built to survive by construction -- which is
the strongest form of "what survives is not what protects".

Run:
  python experiments/analyse_onmanifold.py --t 200
Outputs:
  paper/fig2_onmanifold_vs_additive.png
  experiments/results/onmanifold_summary.md
"""
import io, os, sys, argparse
if not getattr(sys.stdout, "_nvp_utf8", False):   # guard: re-wrapping a second
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # time closes the first wrapper
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from utils.csvio import read_csv as read_csv_safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_ASV = ["WavLM-SV", "ECAPA-TDNN"]          # the two trained ASV systems
ADD_LBL = {"plain_m30_f15": "plain",
           "null_m30_f15": "plain + zero-noise reg.",
           "nsT_m30_f15_l10": "plain + noise-matched reg.",
           "full_m30_f15": "purifier-aware (EOT)",
           "fullnull_m30_f15": "EOT + zero-noise reg."}


def agg(df, keys, t):
    d = df[df.eval_t == t]
    idc = "file" if "file" in d.columns else "speaker"
    surv = d.drop_duplicates(keys + [idc]).groupby(keys).mel_psr.mean()
    prot = d[d.encoder.isin(REAL_ASV)].groupby(keys).defence_valid.mean() * 100
    parts = [surv.rename("survival"), prot.rename("protection")]
    qcols = [c for c in ("pesq", "stoi", "snr_db") if c in d.columns]
    if qcols:   # eval_saved.py records no quality metrics; main_table_v3 does
        parts.append(d.drop_duplicates(keys + [idc]).groupby(keys)[qcols].mean())
    out = pd.concat(parts, axis=1)
    if "pass_through_R" in d.columns:
        out["R_t"] = d.drop_duplicates(keys + [idc]).groupby(keys).pass_through_R.mean()
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=200)
    ap.add_argument("--probe", default=os.path.join(PROJ, "experiments", "results", "parametric_probe.csv"))
    ap.add_argument("--additive", default=os.path.join(PROJ, "experiments", "results", "eval_psr_v3.csv"))
    args = ap.parse_args()

    pr = read_csv_safe(args.probe)
    on = agg(pr, ["transform", "magnitude"], args.t)
    on["label"] = on["transform"] + " " + on["magnitude"].astype(str)

    lines = [f"# On-manifold vs additive protection (attacker: purifier v3 + DDIM, t={args.t})", ""]
    lines += ["## On-manifold transformations (20 held-out speakers)", "",
              "| transform | magnitude | R_t (pass-through) | survival PSR | protection % | PESQ | STOI |",
              "|---|---|---|---|---|---|---|"]
    for _, r in on.sort_values(["transform", "magnitude"]).iterrows():
        lines.append(f"| {r['transform']} | {r.magnitude} | {r.get('R_t', float('nan')):.3f} | "
                     f"{r.survival:.3f} | {r.protection:.0f} | {r.pesq:.2f} | {r.stoi:.3f} |")

    add = None
    if os.path.isfile(args.additive):
        ad = read_csv_safe(args.additive)
        add = agg(ad, ["variant"], args.t)
        add = add[add.variant.isin(ADD_LBL)]
        add["label"] = add.variant.map(ADD_LBL)
        mt = os.path.join(PROJ, "experiments", "results", "main_table_v3.csv")
        if os.path.isfile(mt) and "pesq" not in add.columns:
            q = (read_csv_safe(mt).drop_duplicates(["variant", "file"])
                   .groupby("variant").pesq.mean())
            add["pesq"] = add.variant.map(q)
        lines += ["", "## Additive crafting, same attacker", "",
                  "| method | survival PSR | protection % | PESQ |", "|---|---|---|---|"]
        for _, r in add.iterrows():
            lines.append(f"| {r.label} | {r.survival:.3f} | {r.protection:.0f} | {r.get('pesq', float('nan')):.2f} |")

    # ---- correlation across everything measured under this attacker
    xs = list(on.survival); ys = list(on.protection); fams = ["on-manifold"] * len(on)
    if add is not None:
        xs += list(add.survival); ys += list(add.protection); fams += ["additive"] * len(add)
    xs, ys = np.array(xs), np.array(ys)
    r_all = np.corrcoef(xs, ys)[0, 1]
    r_on = np.corrcoef(on.survival, on.protection)[0, 1] if len(on) > 2 else float("nan")
    lines += ["", f"**Pearson r (survival vs protection), all methods: {r_all:+.2f}**",
              f"  within the on-manifold family alone: {r_on:+.2f}", ""]
    best = on.sort_values("protection", ascending=False).head(3)
    lines += ["Best on-manifold settings by protection:", ""]
    for _, r in best.iterrows():
        lines.append(f"- {r['transform']} {r.magnitude}: protection {r.protection:.0f} %, "
                     f"survival {r.survival:.3f}, PESQ {r.pesq:.2f}")

    out_md = os.path.join(PROJ, "experiments", "results", "onmanifold_summary.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=200)
    cmap = {"tilt": "#2a9d8f", "eq": "#e9c46a", "warp": "#e76f51", "reverb": "#8ecae6"}
    for tname, g in on.groupby("transform", sort=False):
        ax.scatter(g.survival, g.protection, s=46, c=cmap.get(tname, "#888"),
                   label=f"on-manifold: {tname}", zorder=3, edgecolors="white", linewidths=0.6)
    if add is not None:
        ax.scatter(add.survival, add.protection, s=72, c="#264653", marker="D",
                   label="additive crafting", zorder=4, edgecolors="white", linewidths=0.6)
        for _, r in add.iterrows():
            ax.annotate(r.label, (r.survival, r.protection), fontsize=6.5,
                        textcoords="offset points", xytext=(-6, 6), ha="right")
    if len(xs) > 2:
        m, b = np.polyfit(xs, ys, 1)
        gx = np.linspace(xs.min() - .01, xs.max() + .01, 50)
        ax.plot(gx, m * gx + b, "--", c="#b00020", lw=1.4, zorder=2)
    ax.set_xlabel(f"survival through purification (mel PSR at $t^*$={args.t})", fontsize=9)
    ax.set_ylabel("protection after purification (%)", fontsize=9)
    ax.set_title(f"What survives is not what protects   (r = {r_all:+.2f})", fontsize=10)
    ax.grid(alpha=.25, zorder=0); ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=.9)
    fig.tight_layout()
    out_png = os.path.join(PROJ, "paper", "fig2_onmanifold_vs_additive.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    print(f"\nsaved {out_md}\nsaved {out_png}")


if __name__ == "__main__":
    main()
