"""
experiments/calibrate_thresholds.py
===================================
Calibrate the speaker-verification acceptance threshold of every encoder at
the equal-error-rate (EER) operating point on LibriSpeech dev-clean, so the
"clone fails" decision in exp_purifier_aware.py is a real ASV decision rather
than a model-card default.

Protocol: `--n-spk` speakers x `--n-utt` utterances (4-s crops). Every pair of
utterances is a trial (same speaker = target, different = non-target). Cosine
similarity on the REAL evaluation paths of WavLM-SV, ECAPA-TDNN, d-vector.

Run:
  python experiments/calibrate_thresholds.py --n-spk 40 --n-utt 4
Output:
  experiments/results/thresholds.csv   (encoder, eer, threshold, target_mean,
      nontarget_mean, target_sd, nontarget_sd, pooled_sd, dprime, n_target,
      n_nontarget)
  with --dump-scores, also trialscores_<tag>_<encoder>.csv (score, is_target)
"""
import sys, io, os, argparse, itertools
if not getattr(sys.stdout, "_nvp_utf8", False):   # guard: re-wrapping a second
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # time closes the first wrapper
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
import torch
import torch._dynamo  # noqa: F401  (see exp_purifier_aware.py)

from evaluation.evaluate import SAMPLE_RATE
from experiments.exp_purifier_aware import build_encoders, real_embed_all, cs



def eer_threshold(target, nontarget):
    """Threshold where false-accept rate == false-reject rate."""
    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones_like(target), np.zeros_like(nontarget)])
    order = np.argsort(scores)
    scores, labels = scores[order], labels[order]
    n_t, n_n = labels.sum(), (1 - labels).sum()
    # accept iff score >= thr; sweep thr over sorted scores
    fr = np.cumsum(labels) / n_t                      # targets rejected (score < thr)
    fa = 1.0 - np.cumsum(1 - labels) / n_n            # non-targets accepted (score >= thr)
    i = int(np.argmin(np.abs(fr - fa)))
    return float((fr[i] + fa[i]) / 2), float(scores[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spk", type=int, default=40, dest="n_spk")
    ap.add_argument("--n-utt", type=int, default=4, dest="n_utt")
    ap.add_argument("--sec", type=float, default=4.0)
    ap.add_argument("--channel", default="clean",
                    choices=["clean", "gl_only", "phase_only", "diffpure_gl",
                             "diffpure_phase", "diffwave"],
                    help="clean: test utt as-is | gl_only / phase_only: mel->wav "
                         "resynthesis without purification | diffpure_gl / "
                         "diffpure_phase: attacker's DiffPure(t*) then resynthesis "
                         "| diffwave: a FOREIGN waveform purifier, no mel round trip "
                         "(needs --foreign-ckpt; --t-diffuse is ITS timestep)")
    ap.add_argument("--t-diffuse", type=int, default=50, dest="t_diffuse")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--purifier-ckpt", default="", dest="purifier_ckpt")
    ap.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    ap.add_argument("--spk-offset", type=int, default=0, dest="spk_offset",
                    help="skip the first N speakers (purifier v2 training set = 20)")
    ap.add_argument("--cross-session", action="store_true", dest="cross_session",
                    help="FIX S1: draw the enrolment side from a DIFFERENT chapter "
                         "than the test side, so a same-speaker trial does not also "
                         "share a recording session. Drops speakers with only one "
                         "chapter. Without this the EER is optimistic (ours read "
                         "0.03 %% on ECAPA, against ~0.6 %% for comparable work).")
    ap.add_argument("--speaker-set", default="", dest="speaker_set",
                    choices=["", "deantifake", "cross_session"],
                    help="restrict calibration to a named published speaker list, "
                         "so the threshold belongs to the population it will score.")
    ap.add_argument("--with-xvector", action="store_true", dest="with_xvector",
                    help="add SpeechBrain x-vector to the ensemble (appended "
                         "LAST, so existing encoder order is unchanged). Off by "
                         "default: every result before 2026-09-20 was produced "
                         "without it.")
    ap.add_argument("--subset", default="",
                    help="librispeech subset override (default: EVAL_SUBSET). Pin "
                         "it when calibrating for a run on a specific corpus.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump-scores", action="store_true", dest="dump_scores",
                    help="also write the raw target/nontarget trial scores, so "
                         "minDCF, DET curves and bootstrap CIs need no re-run "
                         "(council #19, PLAN_V3 2f)")
    ap.add_argument("--foreign-ckpt", default="", dest="foreign_ckpt",
                    help="DiffWave checkpoint, for --channel diffwave")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.channel + (f"_t{args.t_diffuse}"
                          if args.channel.startswith(("diffpure", "diffwave")) else "")
    if args.sampler != "ddpm":
        tag += "_" + args.sampler
    if args.tag:
        tag += "_" + args.tag
    OUT = os.path.join(PROJ, "experiments", "results", f"thresholds_{tag}.csv")
    diffpure = None
    diffwave = None
    if args.channel.startswith("diffwave"):
        # A waveform-domain purifier needs no mel round trip: the attacker takes
        # the published wav, purifies, and is done. That is a STRONGER attacker
        # than ours, because it never pays the Griffin-Lim cost.
        from models.diffwave_purifier import load_diffwave
        diffwave = load_diffwave(args.foreign_ckpt, t=args.t_diffuse, device=device)
    elif args.channel != "clean":
        from evaluation.evaluate import wav_to_logmel
        from experiments.exp_purifier_aware import attacker_resynth
        voc = "gl" if args.channel.endswith("gl") or args.channel == "gl_only" else "phase"
        if args.channel.startswith("diffpure"):
            from models.real_diffpure import load_real_diffpure
            kw = dict(t_diffuse=args.t_diffuse, device=device, sampler=args.sampler)
            if args.purifier_ckpt:
                kw["checkpoint_path"] = args.purifier_ckpt
            diffpure = load_real_diffpure(**kw)

    from experiments.librispeech import (list_librispeech, load_flac, per_speaker,
                                         cross_session_split, EVAL_SUBSET)
    root = os.path.join(PROJ, "data", "librispeech")
    subset = args.subset or EVAL_SUBSET
    items = list_librispeech(root, subset, min_sec=args.sec)

    # --speaker-set restricts to a NAMED published list. Without this, --n-spk 25
    # would take the first 25 speakers of the subset by id, NOT the 25 a baseline
    # actually published, and the resulting threshold would belong to a different
    # population than the row it is used to score.
    if args.speaker_set:
        from experiments import librispeech as _ls
        keep = set(getattr(_ls, args.speaker_set.upper() + "_SPEAKERS"))
        items = [it for it in items if it["speaker"] in keep]
        print(f"[Calib] speaker-set={args.speaker_set}: "
              f"{len({i['speaker'] for i in items})} of {len(keep)} named speakers")

    # enrol[spk] and test[spk]: the two sides of every trial.
    #
    # FIX S1 (--cross-session). Same-session calibration is what produced our
    # implausible 0.03 % ECAPA EER. `per_speaker` takes the FIRST n_utt files per
    # speaker, and in LibriSpeech those all come from ONE chapter: one sitting, one
    # room, one microphone. A same-speaker trial then shares a recording session, so
    # the encoder is matching the channel as much as the voice and verification
    # looks far easier than it is. VocalBridge, with the same limitation, reports
    # 0.6 % -- twenty times higher than ours.
    #
    # Cross-session draws the enrolment side from a DIFFERENT chapter than the test
    # side, so nothing but the speaker is shared. Speakers with a single chapter
    # cannot support this and are dropped, which is why the speaker count falls.
    if args.cross_session:
        split = cross_session_split(items, min_sec=args.sec)
        spks = sorted(split)[args.spk_offset:args.spk_offset + args.n_spk]
        enrol = {s: split[s]["enroll"][: args.n_utt] for s in spks}
        test = {s: split[s]["ref"][: args.n_utt] for s in spks}
        dropped = len({i["speaker"] for i in items}) - len(split)
        print(f"[Calib] CROSS-SESSION: enrolment and test from different chapters. "
              f"{len(spks)} speakers ({dropped} dropped for having one chapter)")
    else:
        per_spk = per_speaker(items, n_utt=args.n_utt, n_spk=None)
        spks = sorted(per_spk)[args.spk_offset:args.spk_offset + args.n_spk]
        enrol = test = {s: per_spk[s] for s in spks}
        print("[Calib] SAME-SESSION (legacy): enrolment and test may share a "
              "chapter. Expect an optimistic EER; see fix S1.")
    print(f"[Calib] subset={subset} | {len(spks)} speakers x {args.n_utt} utts, "
          f"device={device}")

    encs = build_encoders(device, with_xvector=args.with_xvector)
    names = [e.name for e in encs]

    def embed_clean(it):
        wav = load_flac(it["path"], max_sec=args.sec).to(device)
        return real_embed_all(encs, wav), wav

    def embed_channel(it):
        e, wav = embed_clean(it)
        if diffwave is not None:
            with torch.no_grad():
                return real_embed_all(encs, diffwave(wav, seed=args.seed))
        if args.channel != "clean":
            with torch.no_grad():
                mel = wav_to_logmel(wav, device)
                if diffpure is not None:
                    mel = diffpure(mel, seed=args.seed)
            return real_embed_all(encs, attacker_resynth(mel, wav, voc))
        return e

    embs = {n: [] for n in names}       # enrolment side: always clean
    embs_ch = {n: [] for n in names}    # test side: through the channel
    lab_e, lab_t = [], []
    for spk in spks:
        for it in enrol[spk]:
            e, _ = embed_clean(it)
            for n in names:
                embs[n].append(e[n].cpu())
            lab_e.append(spk)
        for it in test[spk]:
            e_ch = embed_channel(it)
            for n in names:
                embs_ch[n].append(e_ch[n].cpu())
            lab_t.append(spk)
    lab_e, lab_t = np.array(lab_e), np.array(lab_t)

    # Trials pair EVERY enrolment against EVERY test utterance. In cross-session
    # mode the two lists are disjoint by construction, so `a is not b` is enough;
    # in legacy mode they are the same list and a self-pair has to be excluded.
    same_list = not args.cross_session
    rows = []
    for n in names:
        tgt, non = [], []
        # Speaker ids per trial, so a bootstrap can resample SPEAKERS rather
        # than trials (council #19). Trials from one speaker are not
        # independent -- every trial that speaker appears in shares their voice
        # -- so a trial-level bootstrap would report an interval several times
        # too narrow.
        tgt_spk, non_spk = [], []
        for a in range(len(lab_e)):
            for b in range(len(lab_t)):
                if same_list and a == b:
                    continue
                s = cs(embs[n][a], embs_ch[n][b])
                if lab_e[a] == lab_t[b]:
                    tgt.append(s); tgt_spk.append((lab_e[a], lab_t[b]))
                else:
                    non.append(s); non_spk.append((lab_e[a], lab_t[b]))
        tgt, non = np.array(tgt), np.array(non)
        eer, thr = eer_threshold(tgt, non)
        # Fix 2f (PLAN_V3 W1). Only the MEANS were written, so every
        # normalisation question had to infer sigma from the EER under an
        # equal-variance Gaussian assumption that these score distributions do
        # not satisfy. Without the SDs there is no per-encoder scale, so
        # cross-encoder margins are not commensurable (see 2e: the identity of
        # the binding encoder changes for 4 of 29 speakers once margins are
        # divided by each encoder's impostor SD) and council #19 (minDCF, DET,
        # CIs) cannot be answered at all. Zero extra compute: the trial scores
        # are already in hand.
        rows.append(dict(encoder=n, channel=args.channel, eer=round(eer, 4),
                         threshold=round(thr, 4),
                         target_mean=round(tgt.mean(), 4),
                         nontarget_mean=round(non.mean(), 4),
                         target_sd=round(float(tgt.std(ddof=1)), 5),
                         nontarget_sd=round(float(non.std(ddof=1)), 5),
                         # The impostor (nontarget) SD is the scale a margin
                         # should be divided by; the pooled SD is the one a
                         # d-prime style effect size wants.
                         pooled_sd=round(float(np.sqrt(
                             ((len(tgt) - 1) * tgt.var(ddof=1)
                              + (len(non) - 1) * non.var(ddof=1))
                             / max(len(tgt) + len(non) - 2, 1))), 5),
                         dprime=round(float((tgt.mean() - non.mean()) / np.sqrt(
                             0.5 * (tgt.var(ddof=1) + non.var(ddof=1)))), 4),
                         n_target=len(tgt), n_nontarget=len(non),
                         cross_session=bool(args.cross_session), subset=subset,
                         speaker_set=(args.speaker_set or "all")))
        print(f"  {n:12s} EER={100 * eer:.2f}%  thr={thr:.3f}  "
              f"target={tgt.mean():.3f}+-{tgt.std(ddof=1):.3f}  "
              f"nontarget={non.mean():.3f}+-{non.std(ddof=1):.3f}  "
              f"d'={rows[-1]['dprime']:.2f}")
        # Raw trial scores, so minDCF / DET / bootstrap CIs need no re-run
        # (council #19). One small file per (channel, encoder).
        if args.dump_scores:
            safe = "".join(c if c.isalnum() else "_" for c in n)
            sp = os.path.join(os.path.dirname(OUT),
                              f"trialscores_{tag}_{safe}.csv")
            pairs = tgt_spk + non_spk
            pd.DataFrame(dict(
                score=np.concatenate([tgt, non]),
                is_target=np.concatenate([np.ones(len(tgt), dtype=int),
                                          np.zeros(len(non), dtype=int)]),
                spk_enrol=[int(p[0]) for p in pairs],
                spk_test=[int(p[1]) for p in pairs],
            )).to_csv(sp, index=False)
            print(f"    raw trial scores -> {os.path.basename(sp)}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[Calib] saved {OUT}")


if __name__ == "__main__":
    main()
