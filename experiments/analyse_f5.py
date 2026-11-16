"""
experiments/analyse_f5.py
=========================
N2 stage D -- the pre-registered analysis of the F5-TTS column.

PREREG: paper/PREREG_N2_modern_cloner.md (sections 3, 4, 5)

Order of business, and it is the prereg's order:

  1. THE GATE (prereg 4). On `unprotected` / `none`, clone success must hold on
     >= 15 of 29 speakers on at least one encoder, with a positive mean margin
     on that encoder. If it fails, NO protection verdict is printed -- the run
     is reported as "the harness did not establish a working modern cloner".
     This is not a formality: a cloner that fails on clean audio makes every
     protected cell look like a success.

  2. PRIMARY (prereg 3). Paired per-speaker margin, shipping arm MINUS
     unprotected, within (encoder x channel). 9 cells, Holm over the 9,
     paired t on the differences with a BCa interval. Negative = protected.

  3. SECONDARY. The same contrast for the plain arm; success rates with n;
     and the shipping-vs-plain contrast, which is N1 seen through this cloner.

  4. The branch (prereg 5) is printed, but only as the arithmetic -- N2-A needs
     ECAPA plus one other encoder significant in BOTH purifier channels.

Note on the three encoders (analyse_cloning.py's standing caveat): they are
correlated, so "k of 9 significant" is not 9 experiments. Holm is applied
because it is conservative, not because the cells are independent.

Run:
  python experiments/analyse_f5.py
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
from scipy.stats import ttest_rel, wilcoxon

from experiments.analyse_matched_control import bca_ci, holm

DEFAULT_IN = os.path.join(PROJ, "experiments", "results", "cloning_f5tts.csv")
SHIP = "fullnsT_tc_m30_seg"
PLAIN = "plain_tc_m30_seg"
BASE = "unprotected"
ENCODERS = ["ECAPA-TDNN", "WavLM-SV", "Resemblyzer d-vector"]
PRIMARY_ENC = "ECAPA-TDNN"          # the only encoder 42h rates as trustworthy
GATE_MIN = 15                        # prereg 4: >= 15 of 29


def paired_contrast(df, enc, pur, arm, base=BASE):
    """arm MINUS base on the per-speaker margin. Negative = arm protects."""
    s = df[(df.encoder == enc) & (df.purifier == pur)]
    a = s[s.variant == base].set_index("speaker").margin
    b = s[s.variant == arm].set_index("speaker").margin
    j = a.to_frame("base").join(b.to_frame("arm"), how="inner").dropna()
    if len(j) < 6:
        return None
    d = (j.arm - j.base).to_numpy()
    t, p = ttest_rel(j.arm, j.base)
    try:
        _, pw = wilcoxon(j.arm, j.base)
    except ValueError:
        pw = float("nan")
    lo, hi = bca_ci(d)[:2]
    return dict(encoder=enc, purifier=pur, arm=arm, n=len(j),
                delta=float(d.mean()), lo=float(lo), hi=float(hi),
                t=float(t), p=float(p), p_wilcoxon=float(pw),
                lower_in=int((d < 0).sum()),
                base_mean=float(j.base.mean()), arm_mean=float(j.arm.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--ship", default=SHIP)
    ap.add_argument("--plain", default=PLAIN)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if "cloner" in df.columns and df.cloner.nunique() > 1:
        raise SystemExit(f"refusing: more than one cloner in {args.inp}: "
                         f"{sorted(df.cloner.unique())}")
    cloner = df.cloner.iloc[0] if "cloner" in df.columns else "?"
    purs = [p for p in ("none", "ours", "diffwave") if p in set(df.purifier)]
    encs = [e for e in ENCODERS if e in set(df.encoder)]
    print(f"cloner: {cloner}   |   file: {os.path.basename(args.inp)}")
    print(f"channels: {', '.join(purs)}   |   encoders: {', '.join(encs)}")
    print(f"arms: {', '.join(sorted(df.variant.unique()))}")
    print("(margin = clone similarity - threshold; NEGATIVE = clone fails "
          "= victim protected)\n")

    # ---- 1. THE GATE ------------------------------------------------------
    print("=" * 72)
    print("GATE (prereg 4): is this a WORKING cloner on unprotected audio?")
    print("=" * 72)
    gate_pass, gate_lines = False, []
    for e in encs:
        s = df[(df.encoder == e) & (df.purifier == "none") & (df.variant == BASE)]
        if not len(s):
            continue
        k, n = int(s.clone_succeeds.sum()), len(s)
        mm = float(s.margin.mean())
        ok = (k >= GATE_MIN) and (mm > 0)
        gate_pass = gate_pass or ok
        gate_lines.append(f"  {e:<22} {k:>2}/{n:<3} succeed  mean margin "
                          f"{mm:+.4f}   {'PASS' if ok else 'fail'}")
    print("\n".join(gate_lines) if gate_lines else "  (no unprotected/none rows)")
    print(f"\n  GATE: {'PASSED' if gate_pass else 'FAILED'} "
          f"(need >= {GATE_MIN} successes AND positive mean margin "
          f"on at least one encoder)")
    if not gate_pass:
        print("\n" + "!" * 72)
        print("OUTCOME N2-E. The harness did not establish a working modern")
        print("cloner, so NO protection verdict may be read from this run.")
        print("Point estimates below are withheld deliberately (prereg 4;")
        print("the precedent is RESULTS 48, where underpowered cells were")
        print("read as nulls). Debug the harness and re-run ONCE.")
        print("!" * 72)
        return

    # ---- 2. PRIMARY -------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"PRIMARY (prereg 3): {args.ship} MINUS {BASE}, paired per speaker")
    print("=" * 72)
    res = [r for pur in purs for e in encs
           if (r := paired_contrast(df, e, pur, args.ship)) is not None]
    if not res:
        raise SystemExit("no primary cells available")
    padj = holm([r["p"] for r in res])
    for r, pa in zip(res, padj):
        r["p_holm"] = float(pa)
    print(f"{'channel':<10}{'encoder':<22}{'delta':>9}{'95% BCa':>20}"
          f"{'lower':>8}{'p_holm':>10}")
    for r in res:
        star = "***" if r["p_holm"] < .001 else "**" if r["p_holm"] < .01 \
            else "*" if r["p_holm"] < .05 else ""
        print(f"{r['purifier']:<10}{r['encoder']:<22}{r['delta']:>+9.4f}"
              f"  [{r['lo']:+.4f}, {r['hi']:+.4f}]"
              f"{r['lower_in']:>6}/{r['n']:<3}{r['p_holm']:>9.4f} {star}")

    # ---- 3. SECONDARY -----------------------------------------------------
    print("\n" + "=" * 72)
    print(f"SECONDARY: {args.plain} MINUS {BASE} (the naive arm)")
    print("=" * 72)
    res_p = [r for pur in purs for e in encs
             if (r := paired_contrast(df, e, pur, args.plain)) is not None]
    if res_p:
        for r, pa in zip(res_p, holm([r["p"] for r in res_p])):
            star = "*" if pa < .05 else ""
            print(f"{r['purifier']:<10}{r['encoder']:<22}{r['delta']:>+9.4f}"
                  f"  [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                  f"{r['lower_in']:>6}/{r['n']:<3}{pa:>9.4f} {star}")
    else:
        print("  (plain arm not present in this file)")

    # The §59 attribution question, asked of THIS cloner. Worth its own block
    # because the seg-crafted arms have no plain-vs-purifier-aware contrast
    # through DiffVC at all (§59's "not measured" note) -- this is the only
    # place in the project where the SHIPPING recipe is compared with plain.
    print("\n" + "=" * 72)
    print(f"SECONDARY (the §59 attribution, through this cloner):")
    print(f"  {args.ship} MINUS {args.plain}")
    print("  negative = purifier-in-the-loop earns its place")
    print("=" * 72)
    res_s = [r for pur in purs for e in encs
             if (r := paired_contrast(df, e, pur, args.ship,
                                      base=args.plain)) is not None]
    if res_s:
        for r, pa in zip(res_s, holm([x["p"] for x in res_s])):
            star = ("***" if pa < .001 else "**" if pa < .01
                    else "*" if pa < .05 else "")
            print(f"{r['purifier']:<10}{r['encoder']:<22}{r['delta']:>+9.4f}"
                  f"  [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                  f"{r['lower_in']:>6}/{r['n']:<3}{pa:>9.4f} {star}")
        print("\n  NOTE: carries the §59f budget confound -- these two arms")
        print("  share a budget constraint, not a budget spend.")

    print("\n" + "=" * 72)
    print("SECONDARY: clone success rates, with n (RESULTS 31's rule: a rate")
    print("in a thin cell is a range, never a point estimate)")
    print("=" * 72)
    piv = (df.groupby(["purifier", "variant", "encoder"])["clone_succeeds"]
             .agg(k="sum", n="count"))
    piv["rate_%"] = (100 * piv.k / piv.n).round(1)
    print(piv.to_string())

    if PRIMARY_ENC in set(df.encoder):
        s = df[(df.encoder == PRIMARY_ENC) & df.succeeds_at_025.notna()].copy()
        if len(s):
            # The column is written as True/False for ECAPA and empty for the
            # other encoders, so a CSV round trip gives it object dtype and
            # .sum() refuses. Coerce explicitly rather than relying on pandas
            # guessing, which is what broke the first run of this script.
            s["succeeds_at_025"] = (s.succeeds_at_025.astype(str)
                                    .str.strip().str.lower()
                                    .isin({"true", "1", "1.0"}))
            print(f"\nECAPA at the 0.25 field convention:")
            q = s.groupby(["purifier", "variant"])["succeeds_at_025"] \
                 .agg(k="sum", n="count")
            q["rate_%"] = (100 * q.k / q.n).round(1)
            print(q.to_string())

    # ---- 4. THE BRANCH ----------------------------------------------------
    print("\n" + "=" * 72)
    print("BRANCH (prereg 5)")
    print("=" * 72)
    pur_chan = [p for p in purs if p != "none"]
    sig = {(r["encoder"], r["purifier"]) for r in res
           if r["p_holm"] < .05 and r["delta"] < 0}
    helped = [r for r in res if r["p_holm"] < .05 and r["delta"] > 0]
    ecapa_ok = all((PRIMARY_ENC, p) in sig for p in pur_chan) and bool(pur_chan)
    others = [e for e in encs if e != PRIMARY_ENC
              and all((e, p) in sig for p in pur_chan)]
    print(f"  significant protective cells: {len(sig)} of {len(res)}")
    print(f"  ECAPA protective in every purifier channel: {ecapa_ok}")
    print(f"  other encoders likewise: {others or 'none'}")
    if helped:
        print(f"  cells where the protection HELPED the attacker: "
              f"{[(r['encoder'], r['purifier']) for r in helped]}")
    if helped:
        verdict = "N2-D (protection helps the attacker somewhere) -- read with N2-A/B/C below"
    if ecapa_ok and others:
        verdict = "N2-A: protection TRANSFERS to a modern zero-shot cloner"
    elif sig:
        verdict = "N2-B: PARTIAL transfer -- per-encoder result, claim stays DiffVC-scoped"
    else:
        verdict = "N2-C: NO transfer -- the third stated limit (DEFENSE_BRIEF I.4)"
    print(f"\n  => {verdict}")
    print("\n  Reminder: the three encoders are correlated; this is not 9")
    print("  independent experiments. Holm is used because it is conservative.")


if __name__ == "__main__":
    main()
