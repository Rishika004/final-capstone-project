"""
experiments/analyse_mixed.py
============================
The interaction model C2 asserts and the 12-cell grid never tested.

Council v4 E4. R3: "the paper's missing primary analysis, since C2 IS an
interaction claim and the 12-cell grid tests it nowhere."

THE POINT
---------
C2 does not claim "the constraint helps". It claims the constraint's effect
DEPENDS ON THE ENCODER -- it helps WavLM-SV and ECAPA, and (in the matched
condition) reverses on the d-vector. That is a recipe x encoder interaction.

The 12-cell grid tests twelve separate simple effects and then compares their
signs by eye. Twelve one-cell tests can show that some cells are negative and
others positive; they cannot test whether the DIFFERENCE between encoders is
real, which is the actual claim. So:

    margin ~ recipe * encoder * purifier  +  (1 | speaker)

fitted on the per-speaker rows, with speaker as a random intercept because
every cell reuses the same 29 (or 31) speakers -- the non-independence that
RESULTS 48c withdrew the sign p-values over.

WHAT IS AND IS NOT PRE-SPECIFIED
--------------------------------
PLAN_V3 W8 names this model, so the MODEL is pre-specified. Its ROLE is not
confirmatory: `PREREG_05` 4 fixes the confirmatory endpoint as the per-cell
paired difference with Holm over 12, and that stays the result. This is a
PRE-SPECIFIED SECONDARY analysis, and it is reported as one. It cannot rescue a
cell that failed and it is not read for a verdict.

What it legitimately adds, and what the per-cell grid cannot give:
  1. a test of the recipe x encoder INTERACTION itself, with a p-value;
  2. one estimate of the d-vector contrast pooled over purifiers and timesteps,
     with the speaker correlation handled rather than ignored;
  3. the speaker variance component -- how much of the spread is between
     speakers, which is the quantity that decides whether n = 29 was the
     binding constraint.

CODING: treatment contrasts with WavLM-SV as the encoder reference and `full`
as the recipe reference, so the recipe:encoder terms read directly as "how much
more (or less) the constraint does for THIS encoder than for WavLM-SV".
Sign convention is the project's: NEGATIVE margin = better protection, so a
negative `recipe` coefficient means the constraint helps.

Run:
    python experiments/analyse_mixed.py                 # test-clean, 29 spk
    python experiments/analyse_mixed.py --cohort devclean
    python experiments/analyse_mixed.py --cohort heldout
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

R = os.path.join(PROJ, "experiments", "results")
ENC_SHORT = {"WavLM-SV": "wavlm", "ECAPA-TDNN": "ecapa",
             "Resemblyzer d-vector": "dvec"}

COHORTS = {
    # name: (prefix, full tag, fullnsT tag, label)
    "testclean": ("protect_foreign_seg", "full_tc_m30_seg", "fullnsT_tc_m30_seg",
                  "test-clean, segment partials, MATCHED (RESULTS 27)"),
    "devclean": ("dc5_foreign", "full_dc_m30_seg_bpda", "fullnsT_dc_m30_seg_bpda",
                 "dev-clean, segment partials, MATCHED (RESULTS 45)"),
    "heldout": ("heldout_c2", "full_m30_f15_nodvec", "fullnsT_m30_f15_nodvec",
                "test-clean, pooled partials, HELD-OUT (RESULTS 47)"),
    "heldout_seg": ("heldout_seg_c2", "full_m30_seg_nodvec",
                    "fullnsT_m30_seg_nodvec",
                    "test-clean, segment partials, HELD-OUT (RESULTS 53)"),
}
BLOCKS = [("DiffWave-LibriSpeech", 1, "libri_t1"),
          ("DiffWave-LibriSpeech", 3, "libri_t3"),
          ("DiffWave-SC09", 1, "sc09_t1"),
          ("DiffWave-SC09", 3, "sc09_t3")]


def load(prefix, full_tag, nst_tag):
    frames = []
    for pur, t, stem in BLOCKS:
        p = os.path.join(R, f"{prefix}_{stem}.csv")
        if not os.path.isfile(p):
            continue
        d = pd.read_csv(p)
        if "margin" not in d.columns:
            d["margin"] = d.cos_after_purify - d.threshold
        d = d[d.variant.isin([full_tag, nst_tag])].copy()
        if len(d) == 0:
            continue
        d["speaker"] = d.file.astype(str).str.extract(r"^spk(\d+)_")[0]
        d["recipe"] = np.where(d.variant == nst_tag, "fullnsT", "full")
        d["purifier"] = f"{'libri' if 'Libri' in pur else 'sc09'}_t{t}"
        d["enc"] = d.encoder.map(ENC_SHORT)
        frames.append(d[["speaker", "recipe", "purifier", "enc", "margin"]])
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).dropna()
    # one row per (speaker, recipe, purifier, encoder): average the eval seeds
    return (out.groupby(["speaker", "recipe", "purifier", "enc"], as_index=False)
               .margin.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="testclean", choices=sorted(COHORTS))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pre, ft, nt, label = COHORTS[a.cohort]
    out = a.out or os.path.join(R, f"mixed_{a.cohort}.csv")

    d = load(pre, ft, nt)
    if d is None or d.empty:
        sys.exit(f"[mixed] no data for cohort {a.cohort} (prefix {pre})")

    nspk = d.speaker.nunique()
    print("=" * 100)
    print(f"MIXED MODEL -- {label}")
    print(f"margin ~ recipe * enc * purifier + (1 | speaker)      "
          f"n = {len(d)} rows, {nspk} speakers")
    print("NEGATIVE coefficient = better protection. SECONDARY analysis; the")
    print("confirmatory result stays the per-cell grid with Holm over 12.")
    print("=" * 100)

    import statsmodels.formula.api as smf
    d = d.copy()
    d["enc"] = pd.Categorical(d["enc"], categories=["wavlm", "ecapa", "dvec"])
    d["recipe"] = pd.Categorical(d["recipe"], categories=["full", "fullnsT"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_m = smf.mixedlm("margin ~ recipe * enc * purifier", d,
                             groups=d["speaker"]).fit(reml=False)
        # likelihood-ratio test for the recipe x encoder interaction
        red_m = smf.mixedlm("margin ~ recipe + enc * purifier "
                            "+ recipe:purifier", d,
                            groups=d["speaker"]).fit(reml=False)

    res = full_m
    print("\n--- the terms C2 is about (recipe, and recipe x encoder) ---")
    print(f"  {'term':46s} {'coef':>9s} {'95% CI':>20s} {'p':>10s}")
    keep = [p for p in res.params.index
            if p.startswith("recipe") and ":purifier" not in p]
    rows = []
    for p in keep:
        c = res.params[p]
        lo, hi = res.conf_int().loc[p]
        pv = res.pvalues[p]
        print(f"  {p:46s} {c:+9.4f} [{lo:+8.4f},{hi:+8.4f}] {pv:10.2e}")
        rows.append(dict(cohort=a.cohort, term=p, coef=c, lo=lo, hi=hi, p=pv))

    # The simple effect recipe + recipe:enc is read AT THE REFERENCE LEVEL of
    # every other factor -- under treatment contrasts with a 3-way interaction
    # in the model, that is purifier = the alphabetically first level, NOT an
    # average over purifiers. Labelling it "pooled" would be wrong, so both
    # quantities are printed and named.
    ref_pur = sorted(d.purifier.unique())[0]
    base = "recipe[T.fullnsT]"
    print(f"\n--- constraint effect per encoder AT THE REFERENCE PURIFIER "
          f"({ref_pur}) ---")
    print(f"  {'encoder':10s} {'effect':>9s} {'interpretation'}")
    for enc, term in [("wavlm", None), ("ecapa", f"{base}:enc[T.ecapa]"),
                      ("dvec", f"{base}:enc[T.dvec]")]:
        v = res.params.get(base, np.nan)
        if term is not None:
            v = v + res.params.get(term, 0.0)
        verdict = "helps" if v < 0 else "REVERSES"
        print(f"  {enc:10s} {v:+9.4f}  the constraint {verdict} this encoder")
        rows.append(dict(cohort=a.cohort,
                         term=f"simple_effect_{enc}_at_{ref_pur}",
                         coef=v, lo=np.nan, hi=np.nan, p=np.nan))

    # The marginal effect, averaged over the four purifier x t levels. The
    # model is saturated in recipe x enc x purifier, so fitted cell means equal
    # observed cell means and the marginal effect is the mean of the four
    # per-purifier cell effects -- computed from the data, no extrapolation.
    print(f"\n--- constraint effect per encoder, AVERAGED over the four "
          f"purifier x t levels ---")
    piv = d.pivot_table(index=["speaker", "enc", "purifier"],
                        columns="recipe", values="margin")
    piv = piv.dropna()
    cell = (piv["fullnsT"] - piv["full"]).groupby(
        level=["enc", "purifier"]).mean()
    print(f"  {'encoder':10s} {'mean effect':>12s} {'per-purifier spread':>26s}")
    for enc in ["wavlm", "ecapa", "dvec"]:
        if enc not in cell.index.get_level_values("enc"):
            continue
        v = cell.xs(enc, level="enc")
        verdict = "helps" if v.mean() < 0 else "REVERSES"
        print(f"  {enc:10s} {v.mean():+12.4f}   "
              f"[{v.min():+.4f}, {v.max():+.4f}]  {verdict}")
        rows.append(dict(cohort=a.cohort, term=f"marginal_effect_{enc}",
                         coef=float(v.mean()), lo=float(v.min()),
                         hi=float(v.max()), p=np.nan))

    # LR test on the whole recipe x encoder interaction
    lr = 2 * (full_m.llf - red_m.llf)
    df = len(full_m.params) - len(red_m.params)
    from scipy import stats
    p_lr = stats.chi2.sf(lr, df) if df > 0 else np.nan
    print(f"\n--- is the constraint's effect encoder-dependent at all? ---")
    print(f"  likelihood-ratio test on recipe x encoder (and its 3-way terms)")
    print(f"  chi2 = {lr:.2f}, df = {df}, p = {p_lr:.3e}")
    print("  " + ("THE INTERACTION IS REAL -- C2's central claim is supported "
                  "as an interaction," if p_lr < 0.05 else
                  "no detectable interaction: C2's encoder-dependence is NOT "
                  "established by this model,"))
    print("  " + ("not merely as twelve separate cells."
                  if p_lr < 0.05 else "whatever the per-cell signs look like."))
    rows.append(dict(cohort=a.cohort, term="LRT_recipe_x_enc", coef=lr,
                     lo=df, hi=np.nan, p=p_lr))

    # Minimum detectable interaction. RESULTS 48's lesson applies to OUR OWN
    # nulls: a non-significant interaction is only informative against effects
    # the cohort could have detected. The contrast that carries C2 is
    # d-vector vs WavLM, so its SE sets the bound.
    K80 = 2.8016          # z_{.975} + z_{.80}
    term = f"{base}:enc[T.dvec]"
    if term in res.params.index:
        lo_i, hi_i = res.conf_int().loc[term]
        se_i = (hi_i - lo_i) / (2 * 1.959964)
        mde_i = K80 * se_i
        obs = abs(float(res.params[term]))
        print(f"\n--- what interaction could this cohort have detected? ---")
        print(f"  d-vector vs WavLM contrast : {res.params[term]:+.4f}  "
              f"(SE {se_i:.4f})")
        print(f"  MDE at 80 % power          : {mde_i:.4f}")
        print(f"  |observed| / MDE           : {obs / mde_i:.2f}")
        # the matched test-clean contrast is the reference effect size
        REF = 0.1463
        print(f"  the matched test-clean contrast is {REF:+.4f} = "
              f"{REF / mde_i:.2f}x this cohort's MDE")
        if REF > mde_i:
            print("  -> an interaction the size of the matched one WOULD have")
            print("     been detected here. A null is therefore informative")
            print("     against a matched-sized effect.")
        else:
            print("  -> an interaction the size of the matched one would NOT")
            print("     reliably have been detected. A null here is")
            print("     uninformative and must not be read as absence.")
        rows.append(dict(cohort=a.cohort, term="MDE_interaction_dvec",
                         coef=mde_i, lo=se_i, hi=obs, p=np.nan))

    # variance components
    vc = float(res.cov_re.iloc[0, 0]) if res.cov_re.size else np.nan
    rv = float(res.scale)
    print(f"\n--- where the variance is ---")
    print(f"  between-speaker variance : {vc:.5f}  (SD {np.sqrt(vc):.4f})")
    print(f"  residual variance        : {rv:.5f}  (SD {np.sqrt(rv):.4f})")
    icc = vc / (vc + rv) if (vc + rv) > 0 else np.nan
    print(f"  ICC                      : {icc:.3f}")
    print(f"  -> {icc:.0%} of the spread is between speakers, which is why the")
    print(f"     paired per-speaker endpoint is so much more powerful than the")
    print(f"     binary rate, and why n = {nspk} speakers is the binding limit.")
    rows.append(dict(cohort=a.cohort, term="ICC", coef=icc, lo=vc, hi=rv,
                     p=np.nan))

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
