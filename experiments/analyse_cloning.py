"""
experiments/analyse_cloning.py
==============================
Does the protection stop a real VOICE CLONING attack -- and does the
purifier-mismatch finding of section 5 survive in the threat model that actually
motivates the paper?

THREE QUESTIONS, IN ORDER OF WHAT THEY WOULD CHANGE
---------------------------------------------------
Q1  Does protection work against cloning at all?
      protected vs unprotected, per encoder. If this fails, the ASV result does
      not transfer to the real threat and the paper's scope claim is wrong.

Q2  Does it survive the attacker purifying first?
      purifier `none` vs `diffwave`. This is the paper's subject.

Q3  Does the MISMATCH result replicate?
      `fullnsT` vs `full` under a MATCHED purifier (ours, the one crafted against)
      and a MISMATCHED one (DiffWave). Section 5 found the constraint buys nothing
      matched and real protection mismatched. If that pattern reappears here, the
      central finding holds in the cloning threat model rather than only in ASV.

METRIC
------
Similarity between the CLONE and a held-back enrolment utterance of the victim.
Lower = better protection. Statistics on the continuous margin (sim - threshold),
paired by speaker, per encoder: the binary rates saturate and discard the evidence
(RESULTS section 10).

Thresholds are per encoder and EER-calibrated (cosine scales differ wildly: chance
is 0.071 on ECAPA but 0.608 on WavLM-SV, so a single constant is meaningless), with
0.25-on-ECAPA additionally reported as the field convention.

Run:
  python experiments/analyse_cloning.py
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
from scipy.stats import wilcoxon
from utils.csvio import read_csv

R = os.path.join(PROJ, "experiments", "results")
ENC = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]

# ---------------------------------------------------------------------------
# FIX S7 (2026-09-17). The three encoders are NOT three independent tests here.
#
# Found by the F1 ablation: holding Resemblyzer out of CRAFTING collapsed cloning
# protection on all three encoders -- WavLM by 29 points and ECAPA by 44 -- although
# both were still being crafted against. In ASV the encoders verify three separate
# decisions. In cloning they are readouts of ONE upstream event: whether DiffVC's
# conditioning embedding was moved off the victim.
#
# Measured dependence, margin, per (speaker, arm, purifier), n=300:
#     published run   rho = 0.39-0.59   binary agreement 73-82 %
#     held-out run    rho = 0.01-0.43   binary agreement 60-70 %
# So they are strongly dependent but NOT redundant. Both overstatements are wrong:
# "three independent tests" inflates the evidence, and "one measurement reported
# three times" would throw away real signal.
#
# What we do about it: designate ONE primary encoder and report the rest as
# secondary. ECAPA-TDNN is primary, chosen on grounds independent of any result --
# it carries this field's conventional operating point (cosine 0.25) and, unlike
# Resemblyzer, it is NOT the network DiffVC conditions on. We do not apply a
# multiple-comparison correction across encoders, because Holm over a correlated
# family is not the right instrument and would imply an independence we have
# measured to be absent.
PRIMARY_ENC = "ECAPA-TDNN"
# Two perceptual budgets. m30_f15 is the loud one, where protection SATURATES in
# the cloning setting (every arm drives clone success to ~0 %, so no arm can be
# distinguished from another). m20_f20 is tighter and leaves headroom -- ASV
# protection there ranges 5-80 %, which is where a difference between arms can
# actually be resolved. Q3 is only answerable at m20.
BUDGETS = {
    "m30_f15": ("plain_m30_f15", "nsT_m30_f15_l10", "full_m30_f15", "fullnsT_m30_f15_l10"),
    "m20_f20": ("plain_m20_f20", "nsT_m20_f20_l10", "full_m20_f20", "fullnsT_m20_f20_l10"),
}


def load_all(files):
    frames = []
    for f in files:
        p = os.path.join(R, f)
        if os.path.isfile(p):
            frames.append(read_csv(p))
    if not frames:
        raise SystemExit("no cloning results yet")
    return pd.concat(frames, ignore_index=True)


def prefilter(df):
    """Per encoder, keep speakers whose UNPROTECTED audio cloned (AntiFake)."""
    # PER (encoder, purifier). The previous version unioned across purifier
    # conditions -- a speaker who cloned only with NO purifier was then scored under
    # `ours` and `diffwave`, where the unprotected attack had already failed. Those
    # are free protection successes, which is exactly what AntiFake's filter exists
    # to exclude ("given that all samples have already been filtered to bypass SV
    # without AntiFake"). The union inflated the matched-purifier protected rate
    # from 40/50/14 % to 15/21/11 %.
    out = []
    for n in df.encoder.unique():
        for pur in df.purifier.unique():
            ch = df[(df.encoder == n) & (df.purifier == pur)]
            ok = ch[ch.variant == "unprotected"]
            keep = set(ok[ok.clone_succeeds].speaker)
            out.append(ch[ch.speaker.isin(keep)])
    return pd.concat(out, ignore_index=True)


def encoder_dependence(df):
    """Print how correlated the three encoders' decisions are (fix S7).

    Printed every run, not buried in a comment, because the whole per-encoder
    cloning table has to be read in the light of it."""
    import itertools
    d = df.copy()
    piv = d.pivot_table(index=["speaker", "variant", "purifier"],
                        columns="encoder", values="margin").dropna()
    bp = d.pivot_table(index=["speaker", "variant", "purifier"],
                       columns="encoder", values="clone_succeeds").dropna()
    if len(piv) < 10:
        return
    print("=" * 78)
    print("0. HOW INDEPENDENT ARE THE ENCODERS?  (read the tables below in this light)")
    print("=" * 78)
    print(f"  Spearman rho of the margin, and binary agreement, n={len(piv)} cells")
    print("  (computed BEFORE the pre-filter -- see the note at the call site)")
    for a, b in itertools.combinations(sorted(piv.columns), 2):
        rho = piv[a].corr(piv[b], method="spearman")
        agr = 100 * (bp[a] == bp[b]).mean() if (a in bp and b in bp) else float("nan")
        print(f"    {a:<22} vs {b:<22} rho={rho:+.2f}  agree={agr:.0f}%")
    print("")
    print(f"  -> Dependent, but not redundant. `{PRIMARY_ENC}` is the PRIMARY encoder;")
    print("     the other two are secondary and are NOT a family of independent")
    print("     tests. Do not read 'k of 9 cells significant' as 9 experiments.")
    print("")


def paired(d, va, vb, enc):
    s = d[d.encoder == enc]
    a = s[s.variant == va].set_index("speaker").margin
    b = s[s.variant == vb].set_index("speaker").margin
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    if len(j) < 6:
        return None
    try:
        _, p = wilcoxon(j.b, j.a)
    except ValueError:
        return None
    return dict(delta=j.b.mean() - j.a.mean(), better=int((j.b < j.a).sum()),
                n=len(j), p=p)


def row(lbl, r, w=30):
    if r is None:
        print(f"    {lbl:<{w}} (not available)"); return
    star = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else ""
    print(f"    {lbl:<{w}} delta={r['delta']:+.4f}  {r['better']:>2}/{r['n']:<3} "
          f"p={r['p']:.4f} {star}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="m20_f20", choices=list(BUDGETS))
    ap.add_argument("--files", default="")
    a = ap.parse_args()
    PLAIN, NST, FULL, FULLNST = BUDGETS[a.budget]
    files = ([f.strip() for f in a.files.split(",") if f.strip()] or
             (["cloning_protection_m20.csv"] if a.budget == "m20_f20"
              else ["cloning_protection.csv", "cloning_protection_ours.csv"]))
    print(f"budget {a.budget}  |  files: {', '.join(files)}")
    raw = load_all(files)
    # Guard against mixing crafting ensembles. The F1 ablation writes variants
    # tagged `_nodvec` (Resemblyzer HELD OUT) to their own file; if they are ever
    # loaded alongside the white-box tags, the pivot tables below would average a
    # white-box arm with a transfer arm and print one number for two experiments.
    known = set(BUDGETS[a.budget]) | {"unprotected"}
    unknown = sorted(set(raw.variant) - known)
    if unknown:
        raise SystemExit(
            "refusing to analyse: variants present that are not in budget "
            f"{a.budget}: {', '.join(unknown)}. "
            "These are a different crafting ensemble or budget. Analyse them "
            "separately (experiments/analyse_f1.py compares the two runs).")
    df = prefilter(raw)
    purs = [p for p in ("none", "ours", "diffwave") if p in set(df.purifier)]
    print(f"purifier conditions present: {', '.join(purs)}")
    print("(margin = clone similarity - threshold; NEGATIVE = clone fails = protected)\n")
    # On the UNFILTERED frame. The per-(encoder, purifier) pre-filter keeps a
    # different speaker set for each encoder, so the three-way intersection is
    # small AND selected for cells where the encoders already agreed -- measuring
    # dependence there inflates it (rho 0.53-0.67 filtered vs 0.39-0.59 raw).
    raw_m = raw.copy()
    raw_m["margin"] = raw_m.clone_sim - raw_m.threshold
    encoder_dependence(raw_m)

    print("=" * 78)
    print("Q1/Q2. CLONE SIMILARITY by arm (lower = better protection)")
    print("=" * 78)
    for p in purs:
        s = df[df.purifier == p]
        piv = s.pivot_table(index="variant", columns="encoder",
                            values="clone_sim", aggfunc="mean").round(3)
        rate = (100 * s.pivot_table(index="variant", columns="encoder",
                                    values="clone_succeeds", aggfunc="mean")).round(0)
        n_tab = s.pivot_table(index="variant", columns="encoder",
                              values="clone_succeeds", aggfunc="size")
        order = [v for v in ["unprotected", PLAIN, NST, FULL, FULLNST] if v in piv.index]
        print(f"\n--- attacker's purifier: {p} ---")
        print(piv.loc[order].to_string())
        print("\nclone success % (the attack WORKED; lower = better protection):")
        print(rate.loc[order].to_string())
        print("\nn per cell after the per-(encoder,purifier) pre-filter:")
        print(n_tab.loc[order].to_string())
        # Do NOT quote a "best arm". A minimum over four arms at n=5-14 is a biased
        # estimator; report every arm and the mean across the protected ones.
        prot = [v for v in order if v != "unprotected"]
        if prot:
            print("\nMEAN across protected arms (the honest summary):")
            print(rate.loc[prot].mean().round(1).to_string())

    print("\n" + "=" * 78)
    print("Q1. DOES PROTECTION STOP CLONING AT ALL?  (each arm vs unprotected)")
    print("=" * 78)
    for p in purs:
        print(f"\n  purifier: {p}")
        for v in (PLAIN, FULL, FULLNST):
            for e in ENC:
                mark = "  [PRIMARY]" if e == PRIMARY_ENC else "  (secondary)"
                row(f"{v.split('_')[0]:<9} {e}{mark}",
                    paired(df[df.purifier == p], "unprotected", v, e), w=46)

    print("\n" + "=" * 78)
    print("Q3. DOES THE MISMATCH RESULT REPLICATE?   EOT+constraint  vs  EOT alone")
    print("=" * 78)
    print("  section 5 (ASV): nothing under a MATCHED purifier, real gains under a")
    print("  MISMATCHED one. Negative delta = the constraint protects better.")
    for p in purs:
        tag = {"none": "no purifier", "ours": "MATCHED (crafted against)",
               "diffwave": "MISMATCHED (foreign)"}.get(p, p)
        print(f"\n  {p}  --  {tag}")
        for e in ENC:
            mark = "  [PRIMARY]" if e == PRIMARY_ENC else "  (secondary)"
            row(f"{e}{mark}", paired(df[df.purifier == p], FULL, FULLNST, e), w=40)

    print("\n" + "=" * 78)
    print("field convention cross-check (ECAPA only, cos > 0.25)")
    print("=" * 78)
    ec = df[(df.encoder == "ECAPA-TDNN") & df.succeeds_at_025.notna()]
    if len(ec):
        print((100 * ec.pivot_table(index="variant", columns="purifier",
                                    values="succeeds_at_025",
                                    aggfunc="mean")).round(0).to_string())


if __name__ == "__main__":
    main()
