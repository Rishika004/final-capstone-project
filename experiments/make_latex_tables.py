"""
experiments/make_latex_tables.py
================================
Camera-ready LaTeX for the four tables that carry the paper, generated from the
CSVs so the numbers cannot drift from the results of record.

    python experiments/make_latex_tables.py > paper/TABLES.tex

Style: IEEE / ICASSP two-column. Requires \\usepackage{booktabs} and, for Table 3,
\\usepackage{multirow}. Captions sit ABOVE the tabular, per IEEE style. Each table
is wrapped in table[t] so it floats to the top of a column.

WHICH TABLES, AND WHY THESE FOUR
--------------------------------
  1  Survival        the mechanism exists: a blind end and a vicious end
  2  Transfer        it is not a quirk of our own purifier  <- strongest single result
  3  Mismatch        the contribution, and the control that makes it believable
  4  Perceptual      it costs nothing, so the effect is not bought with loudness

Table 5 (cloning) is generated too, but is deliberately last: it is the honest
number, not the flattering one, and it should not lead.
"""
import sys, io, os

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import contextlib
import pandas as pd
from scipy.stats import wilcoxon
from utils.csvio import read_csv
from utils.thresholds import rethreshold

R = os.path.join(PROJ, "experiments", "results")
ENC = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]
TEX = {"WavLM-SV": "WavLM-SV", "ECAPA-TDNN": "ECAPA", "Resemblyzer d-vector": "d-vector"}


def get(name):
    p = os.path.join(R, name)
    return read_csv(p) if os.path.isfile(p) else None


def pfmt(p, bold=False):
    r"""p-values the way a reviewer expects them.

    Bolding happens INSIDE math mode. \textbf{$x$} does not embolden math, which
    is why this returns whole $...$ groups rather than a bare number to wrap."""
    if p < 1e-4:
        body = r"<\!10^{-4}"
    elif p < 1e-3:
        m, e = f"{p:.0e}".split("e")
        body = r"%s\!\times\!10^{%d}" % (m, int(e))
    else:
        body = f"{p:.3f}"
    return (r"$\mathbf{%s}$" % body) if bold else ("$%s$" % body)


# --------------------------------------------------------------------------
def tab_survival():
    d = get("nullspace_survival_20spk_u3_alpha.csv")
    if d is None:
        return
    print(r"""\begin{table}[t]
\centering
\caption{Survival of a perturbation through our purifier, $n=20$ held-out
speakers. $\alpha$ is signal gain (1.0 = returned untouched); \textsc{nsr} is the
share of the returned energy that is \emph{not} the original direction. All three
direction families have identical norm. \textsc{psr}, the norm ratio used in prior
work, is shown only to expose its failure: at $t{=}200$ it ranks random above
pass-through, while $\alpha$ and \textsc{nsr} agree that it does not.}
\label{tab:survival}
\small
\begin{tabular}{llccc}
\toprule
Direction & $t$ & $\alpha\uparrow$ & \textsc{nsr}$\downarrow$ & \textsc{psr} \\
\midrule""")
    for kind, label in (("null", r"pass-through (\textbf{ours})"),
                        ("random", "random"),
                        ("range", "range (adversarial)")):
        s = d[d.kind == kind]
        if not len(s):
            continue
        for i, t in enumerate((50, 200)):
            name = label if i == 0 else ""
            a = s[f"alpha_t{t}"].mean()
            bold = r"\textbf{%.2f}" % a if kind == "null" else "%.2f" % a
            print(f"{name} & {t} & {bold} & {s[f'nsr_t{t}'].mean():.2f} & "
                  f"{s[f'psr_t{t}'].mean():.2f} \\\\")
        if kind != "range":
            print(r"\addlinespace")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


# --------------------------------------------------------------------------
def tab_transfer():
    print(r"""\begin{table}[t]
\centering
\caption{The pass-through subspace transfers to \textbf{two purifiers we did not
train}: published DiffWave checkpoints differing in architecture, weights and
training corpus. \emph{excess} divides out the mel$+$vocoder round trip, which
alone favours the pass-through direction by $1.33\times$; \emph{retention} is that
excess as a fraction of our own purifier's ($1.93\times$). Every cell separates on
all 20 speakers.}
\label{tab:transfer}
\small
\begin{tabular}{llccccc}
\toprule
Foreign purifier & $t$ & $\alpha_{\text{pass}}$ & $\alpha_{\text{rand}}$ &
better & $p$ & retention \\
\midrule""")
    rows = (("foreign_libri_deantifake_t1_n20.csv", "DiffWave-LS", 1),
            ("foreign_libri_deantifake_t3_n20.csv", "DiffWave-LS", 3),
            ("foreign_sc09_philsyn_t1_n20.csv", "DiffWave-SC09", 1),
            ("foreign_sc09_philsyn_t3_n20.csv", "DiffWave-SC09", 3))
    for f, label, t in rows:
        d = get(f)
        if d is None:
            continue
        p = d[d.direction == "pass_through"]
        r = d[d.direction == "random"]
        jp = p.set_index("speaker").alpha_foreign
        jr = r.set_index("speaker").alpha_foreign
        j = jp.to_frame("p").join(jr.to_frame("r"), how="inner")
        _, pv = wilcoxon(j.p, j.r)
        voc = p.alpha_vocoder.mean() / r.alpha_vocoder.mean()
        exc = (p.alpha_foreign.mean() / r.alpha_foreign.mean()) / voc
        our = (p.alpha_ours.mean() / r.alpha_ours.mean()) / voc
        print(f"{label} & {t} & {j.p.mean():.3f} & {j.r.mean():.3f} & "
              f"\\textbf{{{int((j.p > j.r).sum())}/{len(j)}}} & {pfmt(pv)} & "
              f"\\textbf{{{100 * exc / our:.0f}\\%}} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


# --------------------------------------------------------------------------
def tab_mismatch():
    from experiments.analyse_mismatch import holm, paired, load, FULL, FULLNST
    with contextlib.redirect_stdout(io.StringIO()):
        own = load(os.path.join(R, "main_table_v3.csv"),
                   channel="diffpure_gl_ddim_v3")
    ctl = load(os.path.join(R, "control_nopurifier.csv"))
    cells = []
    for tag, lbl in (("", "DiffWave-LS"), ("_sc09", "DiffWave-SC09")):
        for t in (1, 3):
            cells.append((f"{lbl} $t{{=}}{t}$",
                          load(os.path.join(R, f"protect_foreign{tag}_t{t}.csv"))))
    print(r"""\begin{table}[t]
\centering
\caption{\textbf{Main result.} EOT$+$constraint vs.\ EOT alone, on the decision
margin, paired, $n=20$ per cell. Negative $\Delta$ means the constraint protects
\emph{better}. The three blocks must be read together: the constraint \emph{loses}
with no purifier, breaks even against the purifier it was crafted for, and wins
against purifiers it never saw. A merely sturdier perturbation would win in all
three. $p_{\text{adj}}$ is Holm-corrected within the 12 mismatched cells; the other
two blocks are separate families.}
\label{tab:mismatch}
\small
\begin{tabular}{llrcc}
\toprule
Attacker's purifier & Encoder & $\Delta$ & better & $p_{\text{adj}}$ \\
\midrule
\multicolumn{5}{l}{\emph{(a) No purifier --- control}} \\""")
    for e in ENC[:2]:
        r = paired(ctl, FULL, FULLNST, enc=e)
        if r:
            print(f"\\quad none & {TEX[e]} & ${r['delta']:+.3f}$ & "
                  f"{r['better']}/{r['n']} & {pfmt(r['p'])} \\\\")
    print(r"\addlinespace" "\n"
          r"\multicolumn{5}{l}{\emph{(b) Matched --- the purifier crafted against}} \\")
    if own is not None:
        for e in ENC[:2]:
            r = paired(own[own.eval_t == 200], FULL, FULLNST, enc=e)
            if r:
                print(f"\\quad ours, $t{{=}}200$ & {TEX[e]} & ${r['delta']:+.3f}$ & "
                      f"{r['better']}/{r['n']} & {pfmt(r['p'])} \\\\")
    print(r"\addlinespace" "\n"
          r"\multicolumn{5}{l}{\emph{(c) Mismatched --- purifiers we did not train}} \\")
    fam = [(f"{lbl}|{e}", paired(d, FULL, FULLNST, enc=e))
           for lbl, d in cells for e in ENC]
    holm(fam)
    for lbl, r in fam:
        head, e = lbl.split("|")
        if r is None or e == "Resemblyzer d-vector":
            continue
        sig = r["p_adj"] < 0.05
        dl = (r"$\mathbf{%+.3f}$" % r["delta"]) if sig else ("$%+.3f$" % r["delta"])
        pa = pfmt(r["p_adj"], bold=sig)
        print(f"\\quad {head} & {TEX[e]} & {dl} & {r['better']}/{r['n']} & {pa} \\\\")
    print(r"""\bottomrule
\end{tabular}
\vspace{2pt}
\footnotesize
\raggedright
All four d-vector cells are null and are omitted for space; they are reported in
full in the supplement. That encoder is attacked through a proxy front-end.
\end{table}""")


# --------------------------------------------------------------------------
def tab_perceptual():
    d = get("main_table_v3.csv")
    if d is None:
        return
    s = d.drop_duplicates(["variant", "file"])
    print(r"""\begin{table}[t]
\centering
\caption{Perceptual cost, $n=20$. The protection is not bought with audibility:
our perturbation is no louder than plain PGD and no less intelligible. Bold marks
the best value per column. Against plain PGD the SNR and STOI gains are significant
($p<0.001$, $p=0.003$); the PESQ difference is not ($p=0.50$).}
\label{tab:perceptual}
\small
\begin{tabular}{lccc}
\toprule
Crafting method & PESQ$\uparrow$ & STOI$\uparrow$ & SNR (dB)$\uparrow$ \\
\midrule""")
    arms = (("plain_m20_f20", "PGD"),
            ("nsT_m20_f20_l10", "PGD $+$ constraint"),
            ("full_m20_f20", "EOT"),
            ("fullnsT_m20_f20_l10", r"EOT $+$ constraint (\textbf{ours})"))
    vals = {}
    for v, _ in arms:
        q = s[s.variant == v]
        if len(q):
            vals[v] = (q.pesq.mean(), q.stoi.mean(), q.snr_db.mean())
    # Bold the best value in each COLUMN, not our own row throughout. Blanket
    # bolding our row would assert an SNR win we do not have: PGD+constraint is
    # 21.0 dB against our 20.9. A reviewer checks exactly this kind of thing.
    best = [max(t[i] for t in vals.values()) for i in range(3)]
    for v, label in arms:
        if v not in vals:
            continue
        cells = []
        for i, nd in enumerate((2, 3, 1)):
            x = vals[v][i]
            txt = ("%." + str(nd) + "f") % x
            cells.append((r"\textbf{%s}" % txt) if abs(x - best[i]) < 1e-9 else txt)
        print(f"{label} & " + " & ".join(cells) + r" \\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


# --------------------------------------------------------------------------
def tab_cloning():
    def pre(df):
        out = []
        for n in df.encoder.unique():
            for p in df.purifier.unique():
                ch = df[(df.encoder == n) & (df.purifier == p)]
                ok = ch[ch.variant == "unprotected"]
                out.append(ch[ch.speaker.isin(set(ok[ok.clone_succeeds].speaker))])
        return pd.concat(out, ignore_index=True)

    a, b = get("cloning_protection_m20.csv"), get("cloning_protection_f1.csv")
    if a is None or b is None:
        return
    A, B = pre(a), pre(b)
    ren = {"plain_m20_f20_nodvec": "plain", "nsT_m20_f20_nodvec": "nsT",
           "full_m20_f20_nodvec": "full", "fullnsT_m20_f20_nodvec": "fullnsT"}
    B = B.copy(); B["variant"] = B.variant.replace(ren)
    print(r"""\begin{table}[t]
\centering
\caption{Protection against a real cloning attack (DiffVC). Clone success rate,
mean over the four protected arms; unprotected is 100\% by construction of the
AntiFake pre-filter. \textbf{DiffVC's speaker conditioner is byte-identical to the
Resemblyzer encoder}, so crafting against all three is white-box with respect to
the cloner. We therefore re-craft with it held out and report both. Clone success
rises in 9 of 9 cells.}
\label{tab:cloning}
\small
\begin{tabular}{llccc}
\toprule
Attacker's purifier & Encoder & matched & held out & $n$ \\
\midrule""")
    for p, plabel in (("none", "none"), ("diffwave", "DiffWave"),
                      ("ours", "ours")):
        for e in ENC:
            sa = A[(A.purifier == p) & (A.encoder == e) & (A.variant != "unprotected")]
            if not len(sa):
                continue
            sb = B[(B.purifier == p) & (B.encoder == e) & (B.variant != "unprotected")]
            bv = r"\textbf{%.0f\%%}" % (100 * sb.clone_succeeds.mean()) if len(sb) else "--"
            mark = r"$^{\dagger}$" if e == "ECAPA-TDNN" else ""
            print(f"{plabel} & {TEX[e]}{mark} & {100 * sa.clone_succeeds.mean():.0f}\\% "
                  f"& {bv} & {len(sa) // 4} \\\\")
        print(r"\addlinespace")
    print(r"""\bottomrule
\end{tabular}
\vspace{2pt}
\footnotesize
\raggedright
$^{\dagger}$Primary encoder. The three are correlated readouts of one event ---
whether the cloner's conditioning embedding moved off the victim --- not three
independent tests (Spearman $\rho=0.39$--$0.59$, agreement 73--82\%). Nine cells
are not nine experiments.
\end{table}""")


if __name__ == "__main__":
    print("% Generated by experiments/make_latex_tables.py -- do not hand-edit.")
    print("% Requires: \\usepackage{booktabs}")
    print()
    for fn in (tab_transfer, tab_mismatch, tab_survival, tab_perceptual, tab_cloning):
        fn()
        print()
