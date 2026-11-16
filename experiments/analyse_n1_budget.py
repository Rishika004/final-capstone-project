"""
experiments/analyse_n1_budget.py
================================
N1's confound check -- are the plain and purifier-aware arms actually at the
same perceptual budget, and if not, does the difference explain §59c?

WHY THIS EXISTS. §59 reads the purifier-aware arm against the plain arm and
calls them "the same budget" because both were crafted with
`--margin-db 30 --snr-floor 15`. But that is a CONSTRAINT, not a target: an
optimiser facing a harder objective (fool the ensemble THROUGH the purifier)
pushes closer to the constraint boundary than one facing an easy objective.
Equal allowance does not mean equal spend, and §40 measured quality only on the
shipping arm, so nothing in the record had checked this.

It matters because "our arm protects better" and "our arm perturbs more" make
the same prediction for the margin, and a reviewer will say so.

WHAT IS MEASURED
  1. realised spend per arm: waveform SNR against the original 4-s crop, plus
     wideband PESQ, plus RMS level (the §40a loudness confound -- UTMOS and
     NISQA are not level-invariant, so level must be checked separately from
     perturbation energy);
  2. the paired difference in spend between the arms;
  3. TWO tests of whether spend explains the protection gain:
     (a) ACROSS arms -- if magnitude drives it, the arm that spends most should
         gain most. `full_` spends more than `fullnsT_` and can be compared;
     (b) WITHIN speakers -- correlate each speaker's extra dB spent against
         that speaker's protection gain. A magnitude effect predicts a negative
         correlation (more dB spent -> more negative margin delta).

Test (b) with n = 29 only detects a fairly strong correlation, so a null is
evidence, not proof. The clean settlement is a plain arm re-crafted to land at
the purifier-aware arm's realised SNR; that is unmeasured and must not be
asserted. Both caveats are printed.

Run:
  python experiments/analyse_n1_budget.py
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
import soundfile as sf
from scipy.stats import ttest_rel, pearsonr, spearmanr

from experiments.librispeech import load_flac
from experiments.exp_purifier_aware import WAV_DIR

PROMPTS = os.path.join(PROJ, "experiments", "results", "f5_prompts.csv")
ARMS = ["plain_m30_f15_all", "fullnsT_m30_f15_all",
        "plain_m30_f15_nodvec", "full_m30_f15_nodvec", "fullnsT_m30_f15_nodvec"]
PAIRS = [
    ("MATCHED", "cloning_tc_m30_deploy.csv",
     "plain_m30_f15_all", "fullnsT_m30_f15_all"),
    ("HELD-OUT", "cloning_tc_m30_nodvec.csv",
     "plain_m30_f15_nodvec", "fullnsT_m30_f15_nodvec"),
]
ENC = "ECAPA-TDNN"      # the only encoder §42h rates as a trustworthy verifier
PUR = "ours"            # mel DiffPure v7, the channel the claim is about


def spend_table(prompts):
    """Realised perturbation cost per (speaker, arm)."""
    try:
        from pesq import pesq as pesq_fn
    except Exception as e:                                  # noqa: BLE001
        print(f"  (pesq unavailable: {e}; SNR and RMS only)")
        pesq_fn = None
    rows = []
    for _, r in prompts.iterrows():
        o = load_flac(os.path.join(PROJ, r["ref_path"]), 4.0).squeeze(0).numpy()
        for a in ARMS:
            p = os.path.join(WAV_DIR, f"{r['name']}_{a}.wav")
            if not os.path.isfile(p):
                continue
            y, _ = sf.read(p, dtype="float32")
            y = y[: len(o)]
            ref = o[: len(y)]
            d = y - ref
            rows.append(dict(
                speaker=r["speaker"], variant=a,
                snr_db=10 * np.log10((ref ** 2).sum() / max((d ** 2).sum(), 1e-20)),
                pesq=(pesq_fn(16000, ref, y, "wb") if pesq_fn else np.nan),
                rms_db=20 * np.log10(max(np.sqrt((y ** 2).mean()), 1e-12))))
    return pd.DataFrame(rows)


def main():
    global ARMS, PAIRS
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--arms", default="",
                    help="comma-separated arm tags, overriding the N1 set")
    ap.add_argument("--pair", action="append", default=[],
                    help="plain:purifier_aware:LABEL -- repeatable. Overrides "
                         "the N1 pairs. The cloning CSV is looked up only for "
                         "pairs that name one; otherwise only spend is printed.")
    ap.add_argument("--cloning", default="",
                    help="cloning CSV for tests (2a)/(2b); omit to skip them")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.arms:
        ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.pair:
        PAIRS = []
        for spec in args.pair:
            parts = spec.split(":")
            if len(parts) < 3:
                raise SystemExit(f"--pair wants plain:ours:LABEL, got {spec!r}")
            pl, eo, lbl = parts[0], parts[1], ":".join(parts[2:])
            PAIRS.append((lbl, args.cloning, pl, eo))

    prompts = pd.read_csv(args.prompts).drop_duplicates("speaker")
    print("N1 confound check -- realised perceptual spend, and whether it")
    print("explains the generalisation effect of RESULTS 59c.\n")
    q = spend_table(prompts)

    print("Realised spend per arm (LOWER snr_db = spends MORE of the budget):")
    agg = q.groupby("variant")[["snr_db", "pesq", "rms_db"]].agg(["mean", "std"])
    print(agg.round(3).to_string())

    print("\n" + "=" * 72)
    print("(1) Are the arms at equal spend?  purifier-aware MINUS plain")
    print("=" * 72)
    for lbl, _, pl, eo in PAIRS:
        a = q[q.variant == pl].set_index("speaker")
        b = q[q.variant == eo].set_index("speaker")
        j = a.join(b, lsuffix="_p", rsuffix="_e", how="inner")
        for m in ("snr_db", "pesq", "rms_db"):
            if j[m + "_e"].isna().all():
                continue
            d = j[m + "_e"] - j[m + "_p"]
            _, p = ttest_rel(j[m + "_e"], j[m + "_p"])
            flag = "  <- NOT equal" if p < .05 else ""
            print(f"  {lbl:<14}{m:<8}{d.mean():+8.3f}  (sd {d.std():.3f})  "
                  f"p={p:.3g}  n={len(j)}{flag}")

    if args.out:
        q.to_csv(args.out, index=False)
        print(f"\n[budget] spend table -> {args.out}")

    # Tests (2a)/(2b) need a cloning CSV. With --pair and no --cloning there is
    # none, and the spend table above is the whole answer wanted.
    if not all(p[1] for p in PAIRS):
        print("\n(2a)/(2b) skipped: no cloning CSV given for these pairs.")
        print("    The matching check in (1) is what the N7 gate reads.")
        return

    print("\n" + "=" * 72)
    print(f"(2a) ACROSS arms: does the arm that spends MOST gain MOST?")
    print(f"     held-out, {ENC}, channel `{PUR}`")
    print("=" * 72)
    cl = pd.read_csv(os.path.join(PROJ, "experiments", "results",
                                  "cloning_tc_m30_nodvec.csv"))
    s = cl[(cl.encoder == ENC) & (cl.purifier == PUR)]
    base = s[s.variant == "plain_m30_f15_nodvec"].set_index("speaker").margin
    for a in ("full_m30_f15_nodvec", "fullnsT_m30_f15_nodvec"):
        m = s[s.variant == a].set_index("speaker").margin
        j = base.to_frame("p").join(m.to_frame("a"), how="inner")
        snr = q[q.variant == a].snr_db.mean()
        print(f"  {a:<26} SNR {snr:6.2f} dB   delta margin "
              f"{(j.a - j.p).mean():+.4f}")
    print("\n  If magnitude drove the effect the lower-SNR (louder) arm would")
    print("  gain more. Read the two lines above and say which way it goes.")

    print("\n" + "=" * 72)
    print("(2b) WITHIN speakers: extra dB spent vs protection gained")
    print("=" * 72)
    for lbl, cf, pl, eo in PAIRS:
        cl = pd.read_csv(os.path.join(PROJ, "experiments", "results", cf))
        s = cl[(cl.encoder == ENC) & (cl.purifier == PUR)]
        dm = (s[s.variant == eo].set_index("speaker").margin
              - s[s.variant == pl].set_index("speaker").margin).dropna()
        ds = (q[q.variant == pl].set_index("speaker").snr_db
              - q[q.variant == eo].set_index("speaker").snr_db).dropna()
        j = dm.to_frame("dmargin").join(ds.to_frame("extra_dB"), how="inner")
        r, p = pearsonr(j.extra_dB, j.dmargin)
        rs, ps = spearmanr(j.extra_dB, j.dmargin)
        print(f"  {lbl:<9} mean extra spend {j.extra_dB.mean():+.2f} dB, "
              f"mean gain {j.dmargin.mean():+.4f}, n={len(j)}")
        print(f"            pearson r={r:+.3f} (p={p:.3f})   "
              f"spearman={rs:+.3f} (p={ps:.3f})")
        verdict = ("magnitude EXPLAINS the gain" if (p < .05 and r < 0)
                   else "magnitude does NOT explain the gain")
        print(f"            -> {verdict}")

    print("\n" + "=" * 72)
    print("CAVEATS, which belong in the paper beside the numbers")
    print("=" * 72)
    print("  * n = 29 for test (2b): it detects a strong correlation, not a")
    print("    weak one. A null here is evidence, not proof.")
    print("  * the clean settlement is a plain arm RE-CRAFTED to land at the")
    print("    purifier-aware arm's realised SNR. That is unmeasured.")
    print("  * SNR and PESQ are proxies; the budget this project crafts under")
    print("    is psychoacoustic (margin below the masking threshold plus a")
    print("    per-frame SNR floor), so neither is the constraint itself.")
    print("  * RMS is reported separately because UTMOS and NISQA are not")
    print("    level-invariant (the confound 40a caught).")


if __name__ == "__main__":
    main()
