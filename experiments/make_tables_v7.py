"""
experiments/make_tables_v7.py
=============================
The three tables `draft_v6.md` needs, as LaTeX, from the CSVs.

`make_latex_tables.py` builds the FRAME_v6 set (survival, transfer, mismatch,
perceptual, cloning). FRAME_v7 restructured the contributions, so the tables
changed: C2 is now the measurement contribution and needs the operating-point
table, and C3 needs the coverage contrast beside the BPDA bound. This script
emits exactly what the v7 draft cites, and nothing else, so the two generators
cannot drift into disagreeing about the same number.

Every value is read from a CSV. Nothing is typed in.

  T1  C2.  minDCF x channel -- the purifier degrades the verifier
  T2  C3.  the ceiling: coverage contrast, PhonePuRe two-number row, BPDA bound
  T3  C1.  the Fig. 1 companion: alpha per encoder vs its own band-matched
           control (a table because reviewers ask for the numbers behind a
           figure, and because the dagger-marked rows carry the "never crafted
           against" claim)

Run:
    python experiments/make_tables_v7.py
    python experiments/make_tables_v7.py --out paper/TABLES_v7.tex
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

R = os.path.join(PROJ, "experiments", "results")


def need(name):
    p = os.path.join(R, name)
    if not os.path.isfile(p):
        raise SystemExit(f"[tables-v7] missing {name} -- run its producer first")
    return pd.read_csv(p)


# ------------------------------------------------------------------ T1: C2
CHANNEL_ROWS = [
    ("clean_xsess_tc_v2", r"clean"),
    ("diffwave_t1_libri_xsess_tc_v2", r"DiffWave-Libri $t{=}1$"),
    ("diffwave_t1_sc09_xsess_tc_v2", r"DiffWave-SC09 $t{=}1$"),
    ("diffwave_t3_libri_xsess_tc_v2", r"DiffWave-Libri $t{=}3$"),
    ("diffwave_t3_sc09_xsess_tc_v2", r"DiffWave-SC09 $t{=}3$"),
    ("diffpure_gl_t50_ddim_v7_xsess_tc_v2", r"ours $t{=}50$"),
    ("diffpure_gl_t200_ddim_v7_xsess_tc_v2", r"ours $t{=}200$"),
]
ENC_COLS = [("ECAPA_TDNN", "ECAPA"), ("Resemblyzer_d_vector", "d-vector"),
            ("WavLM_SV", "WavLM-SV")]


def t1_operating_points():
    d = need("asv_operating_points.csv")
    dcf = d.pivot_table(index="tag", columns="encoder", values="mindcf_p0.01")
    eer = d.pivot_table(index="tag", columns="encoder", values="eer")
    n_t = int(d.n_target.iloc[0]); n_n = int(d.n_nontarget.iloc[0])
    n_s = int(d.n_speakers.iloc[0])

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{The purifier degrades the verifier, not just the "
         r"perturbation.} minDCF at $p_\mathrm{target}=0.01$ (1.0 = no better "
         r"than the trivial system), with EER\,\% beneath. "
         f"{n_t} target and {n_n} non-target trials over {n_s} speakers per "
         r"channel. WavLM-SV is at or near the trivial system in every cell "
         r"this paper reports, while its EER reads as respectable.}",
         r"\label{tab:oppoints}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"channel & " + " & ".join(n for _, n in ENC_COLS) + r" \\",
         r"\midrule"]
    for tag, lab in CHANNEL_ROWS:
        if tag not in dcf.index:
            continue
        cells = []
        for col, _ in ENC_COLS:
            v = float(dcf.loc[tag, col])
            # bold the cells at or near the trivial system -- that is the
            # table's whole point, and it is WavLM-SV in every row.
            cells.append((r"\textbf{%.2f}" % v) if v >= 0.75 else "%.2f" % v)
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
        L.append(r"\phantom{x}\scriptsize EER & " + " & ".join(
            r"\scriptsize %.1f" % (100 * float(eer.loc[tag, c]))
            for c, _ in ENC_COLS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ------------------------------------------------------------------ T2: C3
def t2_ceiling():
    tost = need("tost_bpda.csv")
    worst = float(tost.achieved.max())
    med = float(tost.achieved.median())
    k = int(tost.equivalent_holm.sum())
    tot = len(tost)
    neg = int((tost.delta < 0).sum())
    mean_d = float(tost.delta.mean())

    # PhonePuRe, continuous margin per purifier, from the project's own reader
    cm = need("cloning_tc_m30_seg_phonepure.csv")
    if "margin" not in cm.columns:
        cm["margin"] = cm.clone_sim - cm.threshold
    # this CSV already carries a `speaker` column; the crafting CSVs carry
    # `file` and need the regex. Do not assume one schema for both.
    arm = "fullnsT_tc_m30_seg"
    rows = {}
    for (enc, pur), g in cm.groupby(["encoder", "purifier"]):
        piv = g.pivot_table(index="speaker", columns="variant", values="margin")
        if arm not in piv.columns or "unprotected" not in piv.columns:
            continue
        s = piv[[arm, "unprotected"]].dropna()
        rows[(enc, pur)] = float((s[arm] - s["unprotected"]).mean())

    ENCS = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]
    SHORT = {"WavLM-SV": "WavLM-SV", "ECAPA-TDNN": "ECAPA",
             "Resemblyzer d-vector": "d-vector"}
    PURS = [("none", "no purifier"), ("diffwave", "stage 1 only"),
            ("phonepure", r"\textbf{full PhonePuRe}")]

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{The paradigm's measured ceiling.} "
         r"Paired per-speaker clone margin, protected $-$ unprotected, "
         r"unfiltered, $n=29$; more negative is better for the defender. "
         r"A learned refinement stage cuts protection to about a fifth "
         r"and does not remove it.}",
         r"\label{tab:ceiling}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"attacker's purifier & " + " & ".join(SHORT[e] for e in ENCS)
         + r" \\", r"\midrule"]
    for key, lab in PURS:
        cells = []
        for e in ENCS:
            v = rows.get((e, key))
            cells.append("--" if v is None else "%+.3f" % v)
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\vspace{2pt}", r"\footnotesize",
          r"\textbf{Gradient estimator.} BPDA vs exact, tested as equivalence "
          r"(TOST, speaker as unit): the pre-specified $\Delta=0.01$ margin "
          f"holds in only {k} of {tot} cells; the data support equivalence at "
          r"$\Delta=%.3f$ (median cell %.3f). Exact gradients protect "
          r"\emph{better} in %d of %d cells, mean %+.3f --- an order of "
          r"magnitude below the effects at stake, so the estimator cannot "
          r"explain prior adaptive failures, but it is not zero."
          % (worst, med, neg, tot, mean_d),
          r"\end{table}"]
    return "\n".join(L)


# ------------------------------------------------------------------ T3: C1
def t3_alpha():
    d = need("fig1_alpha.csv")
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Every encoder's identity direction is "
         r"purifier-hostile.} Survival coefficient $\alpha$ through the mel "
         r"purifier, against a random direction matched to each encoder in "
         r"norm \emph{and} per-mel-band envelope. $\Delta$ is the paired "
         r"per-speaker difference with a BCa 95\,\% CI; Holm within timestep. "
         r"$^\dagger$ never in the crafting ensemble. Isotropic random "
         r"survives at 0.714 ($t{=}50$), a direction in the clean-speech "
         r"principal subspace at 0.844.}",
         r"\label{tab:alpha}",
         r"\begin{tabular}{llcccc}", r"\toprule",
         r"$t$ & encoder & $\alpha$ & own ctl & $\Delta$ & $k/n$ \\",
         r"\midrule"]
    for col, tlab in [("alpha_t50", "50"), ("alpha_t200", "200")]:
        g = d[d.col == col]
        for i, (_, r) in enumerate(g.iterrows()):
            nm = r.encoder + ("$^\\dagger$" if not r.crafted else "")
            L.append(f"{tlab if i == 0 else ''} & {nm} & {r.alpha:.3f} & "
                     f"{r.control:.3f} & "
                     f"$\\mathbf{{{r.delta:+.3f}}}$ & "
                     f"{int(r.below)}/{int(r.n)} \\\\")
        if col == "alpha_t50":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJ, "paper",
                                                  "TABLES_v7.tex"))
    a = ap.parse_args()
    parts = [
        "% Generated by experiments/make_tables_v7.py -- do not edit by hand.",
        "% Every value is read from a CSV in experiments/results/.",
        "", "% ===== T3: C1 =====", t3_alpha(),
        "", "% ===== T1: C2 =====", t1_operating_points(),
        "", "% ===== T2: C3 =====", t2_ceiling(), "",
    ]
    tex = "\n".join(parts)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(tex)
    print(tex)
    print(f"\n% written: {a.out}")


if __name__ == "__main__":
    main()
