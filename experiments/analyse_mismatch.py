"""
experiments/analyse_mismatch.py
===============================
The purifier-mismatch result, assembled from every cell at once.

THE CLAIM UNDER TEST
--------------------
The pass-through constraint is not a protection mechanism in its own right --
sections 8 and 10 established that, and it replicates here against foreign
purifiers. What it appears to be is a TRANSFER mechanism: it buys nothing when
the attacker uses the purifier you crafted against, and real protection when the
attacker uses a different one.

That is what E9 predicts. EOT gradients encode the specific model they were
computed on; the pass-through subspace is shared across purifiers (E9: 86-107 %
retention). So the constraint should be redundant under a matched purifier and
load-bearing under a mismatched one.

THREE COMPARISONS, AND WHY EACH IS NEEDED
-----------------------------------------
  constraint ALONE vs plain        -> the old negative result. Must still fail,
                                      or the story is incoherent.
  EOT+constraint vs EOT, MATCHED   -> our own v3, the purifier crafted against.
                                      Must show nothing.
  EOT+constraint vs EOT, MISMATCHED-> two foreign purifiers. The positive claim.

TWO CONTROLS THAT COULD KILL IT
-------------------------------
  perceptual cost   if the constrained perturbation were simply larger, the gain
                    would be trivial. (It is not: same PESQ, ~1 dB HIGHER SNR.)
  no purifier       if the constrained perturbation were merely sturdier against
                    any channel, it would also win with NO purifier in the loop,
                    and the mismatch framing would be wrong.

Statistics are on the continuous decision margin (cos_after_purify - threshold,
negative = protected), paired by file and encoder. Binarising at the threshold
discards most of the evidence -- see RESULTS section 10, where that change alone
turned p = 0.22-0.63 into p < 0.0001 at zero compute cost. Here the binary rates
are saturated at 95 %, so the margin is the only thing with resolution left.

Run:
  python experiments/analyse_mismatch.py
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
from scipy.stats import wilcoxon
from utils.csvio import read_csv
from utils.thresholds import rethreshold

R = os.path.join(PROJ, "experiments", "results")
ENC = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]
PLAIN, NST = "plain_m30_f15", "nsT_m30_f15_l10"
FULL, FULLNST = "full_m30_f15", "fullnsT_m30_f15_l10"


def load(path, channel=None):
    """Load a results file and, for our own purifier, re-threshold it.

    FIX S2. `exp_purifier_aware.py` stamps ONE `--thresholds` value onto every row
    it writes, so `main_table_v3.csv` carries the t=50 calibration on rows evaluated
    at t=100 and t=200. The matched calibrations were on disk all along and differ by
    up to 0.024 on ECAPA -- the size of the effects we report, and in the defender's
    favour. The foreign and no-purifier files were each run against their own
    calibration and are left alone."""
    if not os.path.isfile(path):
        return None
    d = read_csv(path)
    if channel is not None:
        d = rethreshold(d, channel)
    d["margin"] = d.cos_after_purify - d.threshold
    return d


def holm(results):
    """Holm-Bonferroni over a family of (label, result) pairs; sets r["p_adj"].

    ~60 tests are reported across RESULTS sections 12-18 with no correction
    anywhere, and multiplicity was invoked exactly once -- in section 18b, to
    dismiss a p=0.039 that pointed AGAINST the hypothesis. Correcting only the
    inconvenient result is indefensible, so every confirmatory family gets it."""
    live = [(l, r) for l, r in results if r is not None]
    m = len(live)
    prev = 0.0
    for rank, (l, r) in enumerate(sorted(live, key=lambda x: x[1]["p"])):
        r["p_adj"] = max(min(1.0, r["p"] * (m - rank)), prev)
        prev = r["p_adj"]
    return results


def paired(d, va, vb, enc=None, allow_pooled=False):
    """vb minus va on the margin. Negative delta = vb protects better.

    PSEUDOREPLICATION GUARD (2026-09-17). enc=None pools 3 encoders x 20 files into
    one n=60 Wilcoxon, treating three measurements of the SAME utterance as
    independent. Section 5.1's headline p-values were computed that way; per
    encoder the "significantly harmful" claim rests almost entirely on d-vector --
    the encoder the paper discounts whenever it is null elsewhere. Pass
    allow_pooled=True only if you want the pooled statistic and will label it."""
    if d is None:
        return None
    if enc is None and not allow_pooled:
        raise ValueError(
            "paired() pooled over encoders is pseudoreplication: 3 encoders x the "
            "same 20 files is not n=60. Pass enc=..., or allow_pooled=True and "
            "label the number as pooled.")
    s = d[d.encoder == enc] if enc else d
    a = s[s.variant == va].set_index(["file", "encoder"]).margin
    b = s[s.variant == vb].set_index(["file", "encoder"]).margin
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    if len(j) < 6:
        return None
    try:
        _, p = wilcoxon(j.b, j.a)
    except ValueError:
        return None
    return dict(delta=j.b.mean() - j.a.mean(), better=int((j.b < j.a).sum()),
                n=len(j), p=p)


def row(lbl, r, width=34):
    if r is None:
        print(f"  {lbl:<{width}} (not available)")
        return
    pa = r.get("p_adj")
    ref = pa if pa is not None else r["p"]
    star = "***" if ref < 0.001 else "**" if ref < 0.01 else "*" if ref < 0.05 else ""
    adj = f" p_adj={pa:.4f}" if pa is not None else ""
    print(f"  {lbl:<{width}} delta={r['delta']:+.4f}  "
          f"{r['better']:>2}/{r['n']:<3} p={r['p']:.4f}{adj} {star}")


def main():
    own = load(os.path.join(R, "main_table_v3.csv"),
               channel="diffpure_gl_ddim_v3")
    cells = []
    for tag, lbl in (("", "DiffWave-LibriSpeech"), ("_sc09", "DiffWave-SC09")):
        for t in (1, 3):
            f = os.path.join(R, f"protect_foreign{tag}_t{t}.csv")
            cells.append((f"{lbl} t={t}", load(f)))

    print("=" * 78)
    print("1. THE NEGATIVE RESULT MUST STILL HOLD: constraint ALONE vs plain")
    print("=" * 78)
    print("  (positive delta = the constraint makes protection WORSE)")
    fam = []
    if own is not None:
        for t in sorted(own.eval_t.unique()):
            for e in ENC:
                fam.append((f"our v3 (matched) t={t}  {e}",
                            paired(own[own.eval_t == t], PLAIN, NST, enc=e)))
    for lbl, d in cells:
        for e in ENC:
            fam.append((f"{lbl}  {e}", paired(d, PLAIN, NST, enc=e)))
    for lbl, r in holm(fam):
        row(lbl, r)

    print("\n" + "=" * 78)
    print("2. MATCHED PURIFIER: does the constraint add anything to EOT?")
    print("=" * 78)
    print("  (the purifier the perturbation was crafted against -- expect nothing)")
    fam2 = []
    if own is not None:
        for t in sorted(own.eval_t.unique()):
            for e in ENC:
                fam2.append((f"our v3 t={t}  {e}",
                             paired(own[own.eval_t == t], FULL, FULLNST, enc=e)))
    for lbl, r in holm(fam2):
        row(lbl, r)

    print("\n" + "=" * 78)
    print("3. MISMATCHED PURIFIER: the positive claim")
    print("=" * 78)
    print("  (negative delta = EOT+constraint protects BETTER than EOT alone)")
    # Holm over the whole family of 12 (2 purifiers x 2 strengths x 3 encoders).
    # This is the paper's positive claim, so it gets the same correction as the
    # negative ones -- correcting only inconvenient results is indefensible.
    fam3 = [(f"{lbl}|{e}", paired(d, FULL, FULLNST, enc=e))
            for lbl, d in cells for e in ENC]
    holm(fam3)
    cur = None
    for lbl, r in fam3:
        head, e = lbl.split("|")
        if head != cur:
            print(f"\n  {head}")
            cur = head
        row(f"    {e}", r, width=30)

    print("\n" + "=" * 78)
    print("4. CONTROL: no purifier at all")
    print("=" * 78)
    print("  If the constraint merely made the perturbation sturdier against ANY")
    print("  channel, it would win here too and the mismatch framing would be wrong.")
    ctl = load(os.path.join(R, "control_nopurifier.csv"))
    for e in ENC:
        row(f"  {e}", paired(ctl, FULL, FULLNST, enc=e), width=30)
    if ctl is not None:
        r = paired(ctl, FULL, FULLNST, allow_pooled=True)   # labelled pooled
        # Check the SIGN, not only the p-value. delta = fullnsT - full on the
        # margin, and margin < 0 means protected, so a POSITIVE delta means the
        # constrained perturbation protects WORSE. The first version of this
        # block branched on p alone and therefore announced a large, highly
        # significant DEFICIT as "the constraint helps even with no purifier" --
        # the exact opposite of what the numbers said.
        if r is None:
            pass
        elif r["p"] > 0.05:
            print("\n  -> No difference without a purifier. The effect is specific")
            print("     to purification, which is what the mismatch reading needs.")
        elif r["delta"] > 0:
            print(f"\n  -> With NO purifier the constraint is markedly WORSE "
                  f"(delta={r['delta']:+.3f}, better on {r['better']}/{r['n']}).")
            print("     So this is not generic robustness. The constraint TRADES raw")
            print("     adversarial strength for purification survival: it loses when")
            print("     nothing purifies, breaks even against the purifier it was")
            print("     crafted against, and wins when the attacker's purifier differs.")
        else:
            print("\n  -> The constraint helps even with NO purifier. The effect is")
            print("     GENERIC ROBUSTNESS, not purifier-mismatch insurance. The")
            print("     framing must be corrected.")

    print("\n" + "=" * 78)
    print("5. CONTROL: perceptual cost (is the constrained perturbation just bigger?)")
    print("=" * 78)
    if own is not None:
        s = own[own.variant.isin([FULL, FULLNST])].drop_duplicates(["variant", "file"])
        a = s[s.variant == FULL].set_index("file")
        b = s[s.variant == FULLNST].set_index("file")
        j = a[["pesq", "snr_db", "stoi"]].join(
            b[["pesq", "snr_db", "stoi"]], lsuffix="_f", rsuffix="_fn", how="inner")
        for c in ("pesq", "snr_db", "stoi"):
            _, p = wilcoxon(j[f"{c}_fn"], j[f"{c}_f"])
            d_ = j[f"{c}_fn"].mean() - j[f"{c}_f"].mean()
            note = "(constrained is LESS audible)" if d_ > 0 and c != "pesq" else ""
            print(f"  {c:<8} EOT={j[f'{c}_f'].mean():.3f}  EOT+constraint="
                  f"{j[f'{c}_fn'].mean():.3f}  delta={d_:+.4f}  p={p:.3f} {note}")


if __name__ == "__main__":
    main()
