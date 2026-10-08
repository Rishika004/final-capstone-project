"""Power, minimum detectable effect, and the honest unit of agreement.

Council v4 items B3, B4 and B6, all zero GPU.

WHY THIS EXISTS
---------------
Two sections report "every sign was right" as corroboration:

  * §45 (#5, dev-clean)        12 of 12 cells with the predicted sign
  * §46 (#13b, CAM++/ERes2Net)  8 of 8  cells with the predicted sign

and §46 goes further and reads a NULL off cells that never had the power to
resolve anything.  R3 found three defects in that, and all three are arithmetic
rather than opinion:

  B3  PREREG_13b has no power analysis (PREREG_05 §5 does).  The minimum
      detectable effect at n = 29 is several times the effects actually
      observed, so the test could not have resolved them either way.
  B4  §46 is therefore INCONCLUSIVE, not null.  Accepting a null from a
      low-power test is the error this project criticises elsewhere.
  B6  The p-value on "12 of 12" and "8 of 8" treats cells as independent
      draws.  They are 2-3 factors over the SAME speakers, so the number of
      independent facts is nearer 2-4 than 8-12.  §46e concedes this for its
      own eight cells; §45c does not for its twelve, on the same day.
      The defensible unit is per-speaker agreement WITHIN a cell, which is
      what k/n already reports -- so the fix is to quote k/n and drop the
      across-cell p-value, not to compute a cleverer one.

WHAT IT COMPUTES
----------------
Per cell, from the per-speaker paired differences (not from the summary CSVs,
so the SD is the real one):

    delta     mean paired difference          (the effect)
    sd, se    paired SD and its standard error
    mde80     smallest |effect| this cell could detect at 80 % power,
              two-sided alpha = .05, normal approximation: 2.8016 * se
    power     achieved power AT THE OBSERVED EFFECT, same approximation
    k/n       speakers agreeing with the cell's predicted sign  <- the unit
    ratio     |delta| / mde80 .  Below 1 means a non-significant result in
              this cell carries no information about the hypothesis.

The normal approximation is stated rather than hidden: the confirmatory test is
Wilcoxon, whose efficiency against the t-test is 0.955 for normal data, so the
MDE figures here are if anything slightly optimistic.  They are used only to
label cells informative or uninformative, never to compute a p-value.

Run:
    python experiments/analyse_power.py
    python experiments/analyse_power.py --out experiments/results/power_cells.csv
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

# two-sided alpha = .05, power = .80  ->  z_{1-a/2} + z_{1-b}
Z_ALPHA, Z_POWER = 1.959964, 0.841621
K80 = Z_ALPHA + Z_POWER          # 2.8016


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def power_at(delta, se):
    """Two-sided power at the observed effect, normal approximation."""
    if se <= 0:
        return float("nan")
    lam = abs(delta) / se
    return _norm_cdf(lam - Z_ALPHA) + _norm_cdf(-lam - Z_ALPHA)


def summarise(dd, predicted):
    """dd = per-speaker paired differences; predicted = expected sign."""
    dd = np.asarray(dd, dtype=float)
    n = len(dd)
    sd = float(dd.std(ddof=1))
    se = sd / np.sqrt(n)
    delta = float(dd.mean())
    mde = K80 * se
    agree = int((dd < 0).sum()) if predicted < 0 else int((dd > 0).sum())
    return dict(n=n, delta=delta, sd=sd, se=se, mde80=mde,
                power=power_at(delta, se), ratio=abs(delta) / mde if mde else np.nan,
                agree=agree, predicted=predicted)


# --------------------------------------------------------------- §45 loader
def cells_devclean():
    from experiments.analyse_devclean import (BLOCK_STEMS, ENC_ORDER, FULL,
                                              NST, PRED, margin)
    out = []
    for pur, t, stem in BLOCK_STEMS:
        p = os.path.join(R, stem + ".csv")
        if not os.path.isfile(p):
            continue
        d = margin(pd.read_csv(p))
        d["speaker"] = d.file.astype(str).str.extract(r"^spk(\d+)_")[0].astype(int)
        for enc in ENC_ORDER:
            sub = d[(d.encoder == enc) & (d.variant.isin([FULL, NST]))]
            piv = sub.pivot_table(index="speaker", columns="variant", values="margin")
            if FULL not in piv.columns or NST not in piv.columns:
                continue
            s = piv[[NST, FULL]].dropna()
            row = summarise((s[NST] - s[FULL]).to_numpy(), PRED[(pur, t, enc)])
            row.update(section="45 (#5 dev-clean)", purifier=pur, t=t, encoder=enc)
            out.append(row)
    return out


# --------------------------------------------------------------- §46 loader
def cells_extra():
    p = os.path.join(R, "extra_encoder_protection.csv")
    if not os.path.isfile(p):
        return []
    d = pd.read_csv(p)
    FULL, NST = "full_tc_m30_seg", "fullnsT_tc_m30_seg"
    out = []
    for (enc, pur, t), g in d.groupby(["encoder", "purifier", "t"]):
        piv = g.pivot_table(index="speaker", columns="variant", values="cos_to_clean")
        if FULL not in piv.columns or NST not in piv.columns:
            continue
        s = piv[[NST, FULL]].dropna()
        # §46's endpoint is cosine to the clean speaker; the constraint is
        # predicted to LOWER it, same sign convention as §45's margin.
        row = summarise((s[NST] - s[FULL]).to_numpy(), -1.0)
        row.update(section="46 (#13b unseen encoders)", purifier=pur, t=t,
                   encoder=enc)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "power_cells.csv"))
    a = ap.parse_args()

    cells = cells_devclean() + cells_extra()
    if not cells:
        sys.exit("no cells found")
    df = pd.DataFrame(cells)
    df.to_csv(a.out, index=False)

    for sec, g in df.groupby("section", sort=True):
        print("=" * 104)
        print(f"{sec}   --   can these cells detect the effects they report?")
        print("=" * 104)
        print(f"  {'purifier':22s} {'t':>2} {'encoder':22s} {'delta':>9s} "
              f"{'se':>8s} {'MDE80':>8s} {'|d|/MDE':>8s} {'power':>7s} "
              f"{'k/n':>7s}  informative")
        for _, c in g.iterrows():
            inf = "yes" if c.ratio >= 1.0 else "NO"
            print(f"  {c.purifier:22s} {int(c.t):2d} {c.encoder:22s} "
                  f"{c.delta:+9.4f} {c.se:8.4f} {c.mde80:8.4f} "
                  f"{c.ratio:8.2f} {c.power:7.1%} "
                  f"{c.agree:3d}/{int(c.n):<3d}  {inf}")
        n_un = int((g.ratio < 1.0).sum())
        print(f"\n  {n_un} of {len(g)} cells are below their own 80 %-power "
              f"detectable effect.")
        print(f"  MDE range {g.mde80.min():.4f}-{g.mde80.max():.4f} "
              f"against observed |delta| {g.delta.abs().min():.4f}-"
              f"{g.delta.abs().max():.4f}.")
        print(f"  Median achieved power: {g.power.median():.1%}.")
        print()

    # ------------------------------------------------- B6, the honest unit
    from scipy.stats import binomtest

    from experiments.analyse_matched_control import holm

    print("=" * 104)
    print("B6 -- the unit of agreement is the SPEAKER, not the cell")
    print("=" * 104)
    print("The across-cell counts (12/12, 8/8) treat cells as independent "
          "draws. They are")
    print("2-3 factors over the SAME speakers, so those p-values are dropped. "
          "What replaces")
    print("them: an exact one-sided sign test on per-speaker agreement inside "
          "each cell,")
    print("Holm-corrected within the section's own pre-registered family.\n")

    sign_rows = []
    for sec, g in df.groupby("section", sort=True):
        g = g.reset_index(drop=True)
        praw = [binomtest(int(c.agree), int(c.n), 0.5,
                          alternative="greater").pvalue
                for _, c in g.iterrows()]
        padj = holm(praw)
        n = int(g.n.iloc[0])
        print(f"  {sec}   (chance = {n / 2:.1f} of {n})")
        for (_, c), p, pa in zip(g.iterrows(), praw, padj):
            print(f"    {c.encoder:22s} {c.purifier:22s} t={int(c.t)}  "
                  f"{int(c.agree):3d}/{n:<3d}  p={p:7.4f}  p_holm={pa:7.4f}"
                  f"  {'SIG' if pa < 0.05 else ''}")
            sign_rows.append(dict(section=sec, encoder=c.encoder,
                                  purifier=c.purifier, t=int(c.t),
                                  agree=int(c.agree), n=n, p_sign=p,
                                  p_sign_holm=pa, sign_sig=bool(pa < 0.05)))
        k = sum(1 for x in padj if x < 0.05)
        print(f"    -> {k} of {len(g)} cells show per-speaker agreement above "
              f"chance under Holm")
        print(f"    -> agreement spans {g.agree.min()}-{g.agree.max()} of {n}\n")

    sp = a.out.replace(".csv", "_sign.csv")
    pd.DataFrame(sign_rows).to_csv(sp, index=False)
    print(f"written: {a.out}\nwritten: {sp}")


if __name__ == "__main__":
    main()
