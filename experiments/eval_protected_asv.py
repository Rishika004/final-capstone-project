"""
experiments/eval_protected_asv.py
=================================
Council v4 B7 -- an operating point for the PROTECTED arm. R2's top ask.

THE GAP
-------
RESULTS 42g/42h closed council #19 for eight channels, and every one of them is
a PURIFICATION channel: clean, ours at t = 50/100/200, DiffWave at t = 1/3 on
two checkpoints. So the paper can say what a purifier does to a verifier, and
cannot say what the PROTECTION does in ASV currency -- no EER, no minDCF, no
DET for protected audio. #19 is closed for channels, not for arms.

WHAT THIS MEASURES
------------------
The protected utterance as the TEST side of a verification trial, against the
same clean cross-session enrolments the rest of the paper uses:

    target     protected(spk i)  vs  clean enrolment of spk i
    non-target protected(spk i)  vs  clean enrolment of spk j, j != i

and the same for the unprotected original, so the two are read as a pair. A
protection that works should RAISE the EER of this channel: the system should
stop recognising the speaker from their published audio.

THE HONEST LIMITATION, STATED UP FRONT
--------------------------------------
Only ONE protected utterance exists per speaker (`--utt-index 0`), because
each costs ~5 GPU-min to craft. So this channel has **29 target trials**,
against 464 for every channel in 42h. The non-target side is fine (29 x 28 =
812). Consequences, all of which go in the caption:

  * the EER of this channel is estimated from 29 target scores and its
    interval is correspondingly wide;
  * minDCF at p_target = 0.01 is dominated by the non-target tail, so it is
    better determined than the EER, but still not comparable in precision to
    the 42h table;
  * it may NOT be placed in the same table as the eight purification channels
    without its n beside it, because a reader will compare them.

Crafting three more utterances per speaker would fix this and costs ~7 GPU-h.
That is out of budget before the GPU stop and is recorded as the reason, not
as an oversight.

Run:
    python experiments/eval_protected_asv.py --tag fullnsT_tc_m30_seg
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.analyse_asv import det_curve, eer_from, min_dcf  # noqa: E402
from experiments.librispeech import (cross_session_split,  # noqa: E402
                                     list_librispeech, load_flac,
                                     reference_items)
from utils.provenance import add_prereg_arg, provenance  # noqa: E402

R = os.path.join(PROJ, "experiments", "results")
WAV_DIR = os.path.join(R, "purifier_aware")


def speaker_bootstrap_eer(df, n_boot=2000, seed=20261008, alpha=0.05):
    """Resample SPEAKERS, not trials -- 42e's lesson. Returns (lo, hi) on EER."""
    rng = np.random.default_rng(seed)
    spks = sorted(set(df.spk_test) | set(df.spk_enrol))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(spks, size=len(spks), replace=True)
        sub = pd.concat([df[(df.spk_test == s) | (df.spk_enrol == s)]
                         for s in pick], ignore_index=True)
        if sub.is_target.nunique() < 2:
            continue
        fa, miss, thr = det_curve(sub.score.to_numpy(),
                                  sub.is_target.to_numpy())
        out.append(eer_from(fa, miss, thr)[0])
    if not out:
        return float("nan"), float("nan")
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fullnsT_tc_m30_seg",
                    help="crafted arm whose saved wavs are the test side")
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--utt-index", type=int, default=0, dest="utt_index")
    ap.add_argument("--enroll-utt", type=int, default=1, dest="enroll_utt")
    ap.add_argument("--max-sec", type=float, default=4.0, dest="max_sec")
    ap.add_argument("--out", default=os.path.join(R, "asv_protected_arm.csv"))
    add_prereg_arg(ap)
    a = ap.parse_args()
    PROV = provenance(a.prereg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from experiments.exp_purifier_aware import build_encoders
    encs = build_encoders(device, with_xvector=False)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             a.subset, min_sec=a.max_sec)
    ref = reference_items(items, min_sec=a.max_sec, utt_index=a.utt_index)
    enr = reference_items(items, min_sec=a.max_sec, utt_index=a.enroll_utt)
    ok = set(cross_session_split(items, min_sec=a.max_sec))
    spks = sorted(s for s in ref if s in ok and s in enr)[:a.n_files]
    print(f"[B7] {len(spks)} speakers, arm={a.tag}")

    # ----- embed enrolment (clean, different session) and both test sides
    emb_enr, emb_prot, emb_orig = {}, {}, {}
    missing = []
    for s in spks:
        # The crafted filename carries the utterance index the SHARED
        # REFERENCE SELECTOR actually chose, which is not always --utt-index.
        # Six of 29 speakers use a different one (e.g. spk2830 -> _0005), and
        # hardcoding 0000 silently dropped them, taking an already-thin
        # measurement from 29 target trials to 23 without saying so.
        name = f"spk{s}_{ref[s]['chapter']}_{int(ref[s]['utt']):04d}"
        wp = os.path.join(WAV_DIR, f"{name}_{a.tag}.wav")
        if not os.path.isfile(wp):
            missing.append(os.path.basename(wp))
            continue
        with torch.no_grad():
            we = load_flac(enr[s]["path"], a.max_sec).to(device)
            wo = load_flac(ref[s]["path"], a.max_sec).to(device)
            wp_t = load_flac(wp, a.max_sec).to(device)
            emb_enr[s] = [e.embed(we).detach() for e in encs]
            emb_orig[s] = [e.embed(wo).detach() for e in encs]
            emb_prot[s] = [e.embed(wp_t).detach() for e in encs]
    if missing:
        print(f"[B7] MISSING {len(missing)} crafted wavs, e.g. {missing[:3]}")
    have = sorted(emb_prot)
    if len(have) < 5:
        sys.exit(f"[B7] only {len(have)} speakers usable; aborting")
    print(f"[B7] usable speakers: {len(have)}")

    rows, trials = [], []
    for ei, e in enumerate(encs):
        for arm, bank in (("protected", emb_prot), ("original", emb_orig)):
            sc, tg, se, st = [], [], [], []
            for i in have:
                for j in have:
                    v = float((bank[i][ei] * emb_enr[j][ei]).sum())
                    sc.append(v); tg.append(int(i == j))
                    st.append(i); se.append(j)
            df = pd.DataFrame(dict(score=sc, is_target=tg,
                                   spk_enrol=se, spk_test=st))
            df["encoder"] = e.name; df["arm"] = arm
            trials.append(df)
            fa, miss, thr = det_curve(df.score.to_numpy(),
                                      df.is_target.to_numpy())
            eer, eer_thr = eer_from(fa, miss, thr)
            d01 = min_dcf(fa, miss, thr, 0.01)[0]
            d05 = min_dcf(fa, miss, thr, 0.05)[0]
            lo, hi = speaker_bootstrap_eer(df)
            rows.append(dict(encoder=e.name, arm=arm,
                             n_target=int(df.is_target.sum()),
                             n_nontarget=int((1 - df.is_target).sum()),
                             n_speakers=len(have), eer=round(eer, 4),
                             eer_lo=round(lo, 4), eer_hi=round(hi, 4),
                             eer_threshold=round(eer_thr, 4),
                             mindcf_p001=round(d01, 4),
                             mindcf_p005=round(d05, 4), **PROV))

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.concat(trials, ignore_index=True).to_csv(
        a.out.replace(".csv", "_trials.csv"), index=False)

    print("\n" + "=" * 92)
    print("B7 -- THE PROTECTED ARM AS AN ASV CHANNEL  (R2's top ask)")
    print(f"test side = the published utterance; enrolment = clean, different "
          f"session.  n_target = {rows[0]['n_target']} (NOT 464: one protected "
          f"utterance exists per speaker)")
    print("=" * 92)
    print(f"  {'encoder':22s} {'arm':10s} {'EER %':>8s} {'95% CI':>18s} "
          f"{'minDCF.01':>10s} {'minDCF.05':>10s}")
    for e in encs:
        for arm in ("original", "protected"):
            r = next(x for x in rows if x["encoder"] == e.name
                     and x["arm"] == arm)
            print(f"  {r['encoder']:22s} {arm:10s} {100*r['eer']:8.2f} "
                  f"[{100*r['eer_lo']:6.2f},{100*r['eer_hi']:6.2f}] "
                  f"{r['mindcf_p001']:10.3f} {r['mindcf_p005']:10.3f}")
    print("\n  Protection should RAISE both numbers: the verifier should stop")
    print("  recognising the speaker from their published audio. Read the")
    print("  protected row against the original row on the SAME speakers, not")
    print("  against the 42h table -- that table has 464 target trials and")
    print("  this has %d." % rows[0]["n_target"])
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
