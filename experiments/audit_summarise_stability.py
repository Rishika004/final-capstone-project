"""
experiments/audit_summarise_stability.py
========================================
Tables for RESULTS section 58 from `audit_direction_stability.py`'s output.
Pure pandas; re-run any time.

    python experiments/audit_summarise_stability.py [results/audit_direction_stability_mel29.csv]
"""
import os
import sys

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(PROJ, "experiments", "results")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def rng(s, fmt="{:.3g}"):
    return f"{fmt.format(s.min())} .. {fmt.format(s.max())}" if s.notna().any() else "-"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "audit_direction_stability.csv")
    stem = os.path.splitext(src)[0]
    d = pd.read_csv(src)
    st_path = stem + "_structure.csv"
    if not os.path.isfile(st_path):                      # the 3-speaker run's naming
        st_path = os.path.join(R, "audit_direction_structure.csv")
    st = pd.read_csv(st_path)
    print(f"source {src}")
    spk = sorted(d.speaker.unique())
    print(f"speakers {spk}\n")
    jits = sorted({float(c[3:].split("_")[0]) for c in d.columns if c.startswith("jin") and c.endswith("_cos")})

    for path in ("wav", "mel"):
        g = d[d.path == path]
        if g.empty:
            continue
        print("=" * 110)
        print(f"{path.upper()} PATH -- ranges over {g.speaker.nunique()} speakers")
        print("=" * 110)
        hdr = (f"{'encoder':28s} {'cos@eval':>19s} {'|grad|':>19s} {'|grad| off-max 1e-2':>21s} "
               f"{'repeat':>8s} " + " ".join(f"{'jin'+format(j,'g'):>14s}" for j in jits))
        print(hdr)
        for enc, h in g.groupby("encoder", sort=False):
            line = (f"{enc[:28]:28s} {rng(h.cos_at_eval, '{:.7f}'):>19s} {rng(h.grad_norm):>19s} "
                    f"{rng(h['jev0.01_gn']):>21s} {rng(h.repeat_cos, '{:.4f}'):>8s} ")
            line += " ".join(f"{rng(h[f'jin{j:g}_cos'], '{:+.3f}'):>14s}" for j in jits)
            print(line)
        print("\n  two independent jitter DRAWS at the same magnitude: cosine between the two resulting directions")
        print(f"  {'encoder':28s} " + " ".join(f"{'jin-pair '+format(j,'g'):>16s}" for j in jits)
              + "   " + " ".join(f"{'jev-pair '+format(j,'g'):>16s}" for j in jits))
        for enc, h in g.groupby("encoder", sort=False):
            print(f"  {enc[:28]:28s} " + " ".join(f"{rng(h[f'jin{j:g}_pair'], '{:+.3f}'):>16s}" for j in jits)
                  + "   " + " ".join(f"{rng(h[f'jev{j:g}_pair'], '{:+.3f}'):>16s}" for j in jits))
        if path == "wav" and "abscos_with_rawnorm_grad" in g:
            print("\n  what the x0 residue is: |cos| with grad||u|| (raw-norm gradient) and with v1 (top singular dir.)")
            print(f"  {'encoder':28s} {'|cos(res, grad||u||)|':>22s} {'|cos(res, v1)|':>16s} "
                  f"{'|cos(off-max 1e-2, v1)|':>24s} {'power conv.':>12s}")
            for enc, h in g.groupby("encoder", sort=False):
                print(f"  {enc[:28]:28s} {rng(h.abscos_with_rawnorm_grad, '{:.3f}'):>22s} "
                      f"{rng(h.abscos_with_v1, '{:.3f}'):>16s} {rng(h.abscos_jev_big_v1, '{:.3f}'):>24s} "
                      f"{rng(h.power_last_cos, '{:.4f}'):>12s}")
        gates = [c for c in g.columns if c.startswith("gate_")]
        print("\n  official gate (check_direction_sanity) pass count, by input-jitter magnitude")
        for enc, h in g.groupby("encoder", sort=False):
            print(f"  {enc[:28]:28s} " + "  ".join(f"{c[5:]}: {int(h[c].sum())}/{len(h)}" for c in gates))
        print()

    for path in ("wav-f64", "mel-f64"):
        g = d[d.path == path]
        if g.empty:
            continue
        print("=" * 110)
        print(f"{path.upper()} -- float64 twin on CPU vs the float32 GPU direction")
        print("=" * 110)
        cols = ["speaker", "encoder", "cos_at_eval", "grad_norm_f32", "grad_norm",
                "f64_vs_f32_cos_at_x0"]
        if "f64_vs_f32_cos_offmax" in g and g.f64_vs_f32_cos_offmax.notna().any():
            cols += ["grad_norm_jev1e_2_f32", "grad_norm_jev1e_2_f64", "f64_vs_f32_cos_offmax"]
        print(g[cols].rename(columns={"grad_norm": "grad_norm_f64"}).to_string(index=False,
              float_format=lambda v: f"{v:.4g}"))
        print()

    print("=" * 110)
    print("STRUCTURE -- what equal RMS equalises, and what it does not (means over speakers)")
    print("=" * 110)
    for path in ("mel", "wav"):
        g = st[st.path == path]
        if g.empty:
            continue
        corr = "corr_with_logmel" if path == "mel" else "corr_with_signal_env"
        agg = g.groupby(["kind", "encoder"], sort=False).agg(
            rms=("rms", "mean"), top5=("top5_energy", "mean"), frame_cv=("frame_cv", "mean"),
            corr=(corr, "mean"), kurt=("kurtosis", "mean"), peak_rms=("peak_over_rms", "mean"),
            cos_t=("cos_to_target", "mean")).reset_index()
        print(f"\n{path} domain  (top5 = share of energy in the loudest 5 % of elements; frame_cv = "
              f"CV of per-frame energy; corr = correlation of |d|^2 with the signal's own loudness)")
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
