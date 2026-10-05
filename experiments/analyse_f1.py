"""
experiments/analyse_f1.py
=========================
F1: what does the protection claim look like when Resemblyzer is HELD OUT of
crafting instead of optimised against?

THE PROBLEM THIS ANSWERS
------------------------
DiffVC conditions on a GE2E speaker encoder whose checkpoint is byte-identical to
the Resemblyzer encoder in our crafting ensemble. So every published cloning number
is a MATCHED-encoder number: the perturbation was optimised against the exact
network the cloner reads identity with. That is the strongest possible setting and
the least informative one. AntiFake's Table 4 is the prior for how much it matters,
and it is a cliff, not a slope: single-encoder crafting protects the synthesizer
built on that encoder at 99.6-100 % and the other four at 0-20.6 %. Their GE2E row
is our case exactly -- 100 % on SV2TTS, which uses GE2E, against 1.2-20.6 % on the
four that do not. (Caveat carried from the source: that table varies the crafting
ensemble and evaluates on SYNTHESIZERS. It never holds an encoder out and evaluates
on it, so it is a prior for encoder match mattering, NOT prior art for the
leave-one-out protocol used here.)

Re-crafting with `--craft-encoders no-dvec` makes Resemblyzer a held-out encoder.
Everything else is held fixed (corpus, speakers, purifier, thresholds, budget,
steps), so the difference between the two runs is attributable.

WHAT EACH OUTCOME MEANS -- decided BEFORE looking
-------------------------------------------------
  d-vector protection survives the drop
        The cloning claim is a transfer result. It gets STRONGER: protection
        crafted against two encoders still defeats a cloner reading a third.
  d-vector protection collapses, WavLM/ECAPA hold
        The ASV claim stands; the CLONING claim was white-box and must be
        restated as such. This is the outcome AntiFake's Table 4 predicts.
  everything collapses
        The two dropped-encoder gradients were carrying the whole attack, and the
        ensemble was doing less work than claimed.

None of these is a failure of the experiment. The failure mode was NOT running it
and letting a reviewer discover the shared checkpoint.

Run:
  python experiments/analyse_f1.py
"""
import sys, io, os

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import pandas as pd
from scipy.stats import wilcoxon
from utils.csvio import read_csv
from utils.thresholds import rethreshold

R = os.path.join(PROJ, "experiments", "results")
ENC = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]
HELD_OUT = "Resemblyzer d-vector"
ARMS = [("plain_m20_f20", "plain_m20_f20_nodvec"),
        ("nsT_m20_f20_l10", "nsT_m20_f20_nodvec"),
        ("full_m20_f20", "full_m20_f20_nodvec"),
        ("fullnsT_m20_f20_l10", "fullnsT_m20_f20_nodvec")]


def load(name, channel=None):
    p = os.path.join(R, name)
    if not os.path.isfile(p):
        return None
    d = read_csv(p)
    return rethreshold(d, channel) if channel else d


def margin(d):
    """Both runs are re-thresholded per eval_t before this (fix S2), so the two
    sides of the comparison sit on the same calibration."""
    d = d.copy()
    d["margin"] = d.cos_after_purify - d.threshold
    return d


def unpaired_note(n_a, n_b):
    return "" if n_a == n_b else f"  [!] unequal n: {n_a} vs {n_b}"


def asv():
    print("=" * 78)
    print("1. ASV: crafted against 3 encoders  vs  crafted against 2 (d-vector held out)")
    print("=" * 78)
    a = load("main_table_v3.csv", channel="diffpure_gl_ddim_v3")
    b = load("main_table_f1.csv", channel="diffpure_gl_ddim_v3")
    if a is None or b is None:
        print("  (main_table_f1.csv not written yet -- run experiments/run_f1_nodvec.sh)")
        return
    a, b = margin(a), margin(b)
    print("  margin = cos_after_purify - threshold. NEGATIVE = protected.")
    print("  delta = held-out run minus published run. POSITIVE = protection LOST.")
    for et in sorted(set(a.eval_t) & set(b.eval_t)):
        print(f"\n  --- attacker purification t={et} ---")
        for va, vb in ARMS:
            for e in ENC:
                sa = a[(a.variant == va) & (a.encoder == e) & (a.eval_t == et)]
                sb = b[(b.variant == vb) & (b.encoder == e) & (b.eval_t == et)]
                if len(sa) == 0 or len(sb) == 0:
                    continue
                j = (sa.set_index("file").margin.to_frame("pub")
                     .join(sb.set_index("file").margin.to_frame("held"), how="inner"))
                if len(j) < 6:
                    continue
                try:
                    _, p = wilcoxon(j.held, j.pub)
                except ValueError:
                    p = float("nan")
                d = j.held.mean() - j.pub.mean()
                flag = "  <-- HELD OUT" if e == HELD_OUT else ""
                star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                print(f"    {va:<22}{e:<22} pub={j.pub.mean():+.3f} "
                      f"held={j.held.mean():+.3f}  delta={d:+.4f} n={len(j)} "
                      f"p={p:.4f} {star}{flag}")

    print("\n  protection RATE (margin < 0), for the binary reading:")
    for et in sorted(set(a.eval_t) & set(b.eval_t)):
        for e in ENC:
            ra = 100 * (a[(a.encoder == e) & (a.eval_t == et)
                          & a.variant.isin([x for x, _ in ARMS])].margin < 0).mean()
            rb = 100 * (b[(b.encoder == e) & (b.eval_t == et)
                          & b.variant.isin([y for _, y in ARMS])].margin < 0).mean()
            flag = "  <-- HELD OUT" if e == HELD_OUT else ""
            print(f"    t={et:<4} {e:<22} published={ra:5.1f}%  held-out={rb:5.1f}%"
                  f"  ({rb - ra:+.1f} pts){flag}")


def prefilter(df):
    """Per (encoder, purifier), keep speakers whose UNPROTECTED audio cloned."""
    out = []
    for n in df.encoder.unique():
        for pur in df.purifier.unique():
            ch = df[(df.encoder == n) & (df.purifier == pur)]
            ok = ch[ch.variant == "unprotected"]
            out.append(ch[ch.speaker.isin(set(ok[ok.clone_succeeds].speaker))])
    return pd.concat(out, ignore_index=True) if out else df


def cloning():
    print("\n" + "=" * 78)
    print("2. CLONING: the number that actually decides the paper's claim")
    print("=" * 78)
    a, b = load("cloning_protection_m20.csv"), load("cloning_protection_f1.csv")
    if a is None or b is None:
        print("  (cloning_protection_f1.csv not written yet)")
        return
    ren = dict(ARMS)
    a, b = prefilter(a), prefilter(b)
    b = b.copy()
    b["variant"] = b.variant.replace({v: k for k, v in ren.items()})
    print("  clone success % -- the attack WORKED. Lower = better protection.")
    print("  The d-vector column is the one to read: matched encoder in the")
    print("  published run, HELD OUT in the F1 run, and it is the encoder DiffVC")
    print("  itself conditions on.\n")
    for pur in [p for p in ("none", "ours", "diffwave") if p in set(a.purifier)]:
        print(f"  --- attacker's purifier: {pur} ---")
        for e in ENC:
            sa = a[(a.purifier == pur) & (a.encoder == e) & (a.variant != "unprotected")]
            sb = b[(b.purifier == pur) & (b.encoder == e) & (b.variant != "unprotected")]
            if len(sa) == 0 or len(sb) == 0:
                print(f"    {e:<22} (missing)"); continue
            ra, rb = 100 * sa.clone_succeeds.mean(), 100 * sb.clone_succeeds.mean()
            flag = "  <-- HELD OUT in F1" if e == HELD_OUT else ""
            print(f"    {e:<22} published={ra:5.1f}% (n={len(sa)})   "
                  f"held-out={rb:5.1f}% (n={len(sb)})   {rb - ra:+5.1f} pts{flag}")
        print()

    print("  NOTE ON n. The per-(encoder, purifier) pre-filter leaves single-digit")
    print("  cells. A shift of 10-20 points here is inside the binomial interval.")
    print("  Read the DIRECTION and whether it is consistent across purifiers;")
    print("  do not quote a difference of a few points as an effect.")


if __name__ == "__main__":
    asv()
    cloning()
