"""
experiments/exp_cloning_protection.py
=====================================
2.1 -- Does the protection actually stop a VOICE CLONING attack?

Every protection number in this project so far is speaker verification: we measure
whether the purified audio still matches the victim. That is a proxy. The threat is
cloning, and a cloning model could either absorb a perturbation that fools ASV
directly, or amplify it. Nobody can tell which from our data.

THE ATTACKER'S FULL CHAIN
-------------------------
    victim publishes      x_p = x + delta        (protected)
    attacker purifies     x_r = Purify(x_p)      (optional; the paper's subject)
    attacker clones       y = DiffVC(source=own speech, reference=x_r)
    verifier asks         is cos(E(y), E(enrolment)) >= tau ?

`reference` is the victim's published clip -- protection is applied THERE, never to
the source. `enrolment` is a THIRD, held-back utterance of the victim, so a
"successful clone" cannot be the trivial act of matching one file to itself.

THE PRE-FILTER, AND WHY IT IS NOT OPTIONAL
------------------------------------------
AntiFake (CCS 2023) evaluates only on samples that cloned successfully WITHOUT
protection: *"AERR is defined as the rate at which DeepFake samples can no longer
bypass the corresponding SV system, given that all samples have already been
filtered to bypass SV without AntiFake."* Without that filter, a speaker whose
clean audio never cloned would be scored as a protection success. Our numbers would
be inflated and incomparable to theirs. `gate_cloning.py` already showed 8/10 clean
clones succeed, so the filter removes a real but small minority.

THRESHOLDS -- PER ENCODER, and the 0.25 convention is ECAPA-ONLY
----------------------------------------------------------------
The field convention is **cosine 0.25 on ECAPA-TDNN** (SafeSpeech, VoiceCloak,
ClearMask; De-AntiFake uses 0.25 on x-vector), from Jiang et al., Mega-TTS 2,
ICLR 2024. We report it for ECAPA so our numbers are comparable.

**It must not be applied to the other encoders.** Cosine scales differ per encoder:
measured chance floors (different-speaker pairs) are 0.071 for ECAPA but 0.549 for
Resemblyzer d-vector and 0.608 for WavLM-SV. A 0.25 threshold sits BELOW chance for
those two, so every clone trivially "succeeds" and the rate carries no information.
An earlier version of this script did exactly that and reported 100 % success on
d-vector and WavLM for every arm, protected or not.

So: each encoder gets its own EER-calibrated threshold from the clean channel
(`thresholds_clean.csv`, where the enrolment lives), and ECAPA additionally gets the
0.25 convention. The continuous margin is reported regardless, because binarising
discards most of the evidence (RESULTS section 10).

Run:
  python experiments/exp_cloning_protection.py --n-files 20 --purifiers none,diffwave
"""
import sys, io, os, argparse, time, zlib

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
import torch
import torch._dynamo   # noqa: F401
import transformers    # noqa: F401
import soundfile as sf

from experiments.exp_purifier_aware import build_encoders, real_embed_all, cs, WAV_DIR
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET
from evaluation.cloning_attack import FIELD_CONVENTION_THRESHOLD
from utils.csvio import read_csv
from utils.provenance import provenance, add_prereg_arg

OUT = os.path.join(PROJ, "experiments", "results", "cloning_protection.csv")
THR_CLEAN = os.path.join(PROJ, "experiments", "results", "thresholds_clean.csv")


def load_thresholds(path=None):
    """Per-encoder EER thresholds calibrated on the clean channel.

    `path` overrides the default file. Under --cross-session the run must use the
    cross-session calibration, or the clones are scored at an operating point
    derived from a different (and easier) speaker population.

    WHY THIS DOES *NOT* VARY BY THE ATTACKER'S PURIFIER (checked for fix S2,
    2026-09-17, and deliberately left unchanged).

    FIXES_TO_PUBLISHABLE S2 claimed this was the same bug as the main table's
    threshold reuse. It is not. In the ASV experiments the protected audio itself
    passes through the purifier before reaching the encoder, so the purifier is part
    of the channel and the threshold must match it. Here the scored trial is

        clone = DiffVC(source, Purify(reference))   vs   enrolment (clean)

    The purifier acts on the REFERENCE, upstream of the cloner. The clone that the
    verifier sees is DiffVC output either way, and the enrolment is clean either way,
    so the verifier's channel is identical across the `none` / `ours` / `diffwave`
    conditions. Varying the threshold with the attacker's purifier would mean the
    verifier somehow knows which purifier the attacker used, which is not the threat
    model -- and it would make the three conditions incomparable.

    What IS a real caveat, and is not fixable by swapping calibration files: the
    calibration trials are natural-vs-natural while the scored trials are
    synthetic-vs-natural. We keep the clean-channel threshold because it models a
    deployed verifier calibrated on real speech, which is the deployed case. State it
    as a caveat, do not "fix" it by calibrating on clones -- that would treat clones
    as a legitimate enrolment class and define the attack away.
    """
    path = path or THR_CLEAN
    if not os.path.isfile(path):
        raise SystemExit(
            f"missing {os.path.basename(path)} -- run: python "
            "experiments/calibrate_thresholds.py --channel clean "
            "--cross-session --subset test-clean --n-spk 29 --n-utt 4")
    d = read_csv(path)
    return {r.encoder: float(r.threshold) for r in d.itertuples()}


DW_LIBRI = os.path.join(PROJ, "checkpoints", "foreign", "diffwave_libri_deantifake.pkl")


def build_purifiers(names, device, dw_t=1, ours_t=200, ours_ckpt=None):
    """name -> callable(wav16)->wav16. 'none' is the identity."""
    out = {}
    for n in names:
        if n == "none":
            out[n] = lambda w: w
        elif n == "phonepure":
            # RESULTS 29: De-AntiFake's full PhonePuRe (stage 1 + 2) is run OFFLINE
            # by their own code (external/De-AntiFake, prep_phonepure_inputs.py), so
            # this "purifier" is a lookup of the pre-purified file, resolved per
            # reference in the main loop. Sentinel: None.
            out[n] = None
        elif n == "diffwave":
            from models.diffwave_purifier import load_diffwave
            dw = load_diffwave(DW_LIBRI, t=dw_t, device=str(device))
            out[n] = lambda w, dw=dw: dw(w, seed=1)
        elif n == "ours":
            from models.real_diffpure import load_real_diffpure, MelDiffPure
            from evaluation.evaluate import wav_to_logmel
            from utils.vocoder import invert_mel
            dp = load_real_diffpure(checkpoint_path=ours_ckpt or os.path.join(
                                        PROJ, "checkpoints", "purifier_v3.pt"),
                                    t_diffuse=ours_t, device=device, sampler="ddim")
            loop = MelDiffPure(dp.denoiser, n_steps=dp.n_steps, t_diffuse=ours_t,
                               sampler="ddim").to(device).eval()
            def _f(w, loop=loop, device=device):
                with torch.no_grad():
                    return invert_mel(loop(wav_to_logmel(w, device), seed=1).cpu()).to(w.device)
            out[n] = _f
        else:
            raise SystemExit(f"unknown purifier {n}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=20, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--ref-utt", type=int, default=3, dest="ref_utt")
    ap.add_argument("--enroll-utt", type=int, default=0, dest="enroll_utt")
    ap.add_argument("--variants",
                    default="plain_m30_f15,nsT_m30_f15_l10,full_m30_f15,fullnsT_m30_f15_l10")
    ap.add_argument("--purifiers", default="none,diffwave")
    ap.add_argument("--speaker-set", default="", dest="speaker_set",
                    choices=["", "deantifake", "cross_session"],
                    help="restrict to a named published speaker list so this row "
                         "can sit beside that paper's table.")
    ap.add_argument("--cross-session", action="store_true", dest="cross_session",
                    help="FIX S1: enrolment from a DIFFERENT chapter than the "
                         "reference, and the same speaker set the cross-session "
                         "thresholds were calibrated on.")
    ap.add_argument("--subset", default="",
                    help="librispeech subset override (default: EVAL_SUBSET, now "
                         "test-clean). Pin to dev-clean to score perturbations "
                         "crafted there; the saved wav names encode the corpus.")
    ap.add_argument("--dw-t", type=int, default=1, dest="dw_t")
    ap.add_argument("--thresholds", default="",
                    help="threshold CSV to score against; must match the speaker "
                         "population and channel of this run")
    ap.add_argument("--ours-ckpt", default="", dest="ours_ckpt",
                    help="purifier checkpoint for the 'ours' condition "
                         "(default: purifier_v3.pt)")
    ap.add_argument("--save-clones", action="store_true", dest="save_clones",
                    help="write each generated clone to "
                         "results/purifier_aware/clones/ so the ATTACKER's "
                         "output can be quality-scored (council #6). ~22 MB for "
                         "a 29-speaker run.")
    ap.add_argument("--clone-seed", type=int, default=None, dest="clone_seed",
                    help="seed DiffVC's diffusion sampler per (speaker, arm, "
                         "purifier) so this run is reproducible. Default off: "
                         "every cloning CSV on disk was written unseeded and "
                         "passing a seed would change those numbers.")
    ap.add_argument("--out", default=OUT)
    add_prereg_arg(ap)
    args = ap.parse_args()
    PROV = provenance(args.prereg)   # code_sha / prereg_sha / argv per row (W0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [v for v in args.variants.split(",") if v]
    pur_names = [p for p in args.purifiers.split(",") if p]

    THR = load_thresholds(args.thresholds or None)
    print("per-encoder EER thresholds (clean channel): " +
          ", ".join(f"{k}={v:.3f}" for k, v in THR.items()))
    from models.diffvc_clone import DiffVCCloner
    vc = DiffVCCloner(device=str(device))
    encs = build_encoders(device)
    purs = build_purifiers(pur_names, device, dw_t=args.dw_t,
                           ours_ckpt=args.ours_ckpt or None)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             args.subset or EVAL_SUBSET, 4.0)

    # FIX S1 applies here too, and arguably matters more than it does for ASV.
    # The verdict is cos(clone, enrolment). With same-chapter selection the
    # enrolment shared a recording session with the reference the clone was built
    # from, so the comparison was flattered by channel match on BOTH sides. Under
    # --cross-session the enrolment comes from a different chapter entirely, and
    # the speaker set matches the one the thresholds were calibrated on.
    if args.speaker_set:
        from experiments import librispeech as _ls
        keep = set(getattr(_ls, args.speaker_set.upper() + "_SPEAKERS"))
        items = [it for it in items if it["speaker"] in keep]
        print(f"[clone] speaker-set={args.speaker_set}: "
              f"{len({i['speaker'] for i in items})} of {len(keep)} named speakers")

    if args.cross_session or args.speaker_set:
        # SAME shared selector the crafting script uses, so the wav we ask for is
        # the wav that was made. Enrolment is the part --cross-session governs.
        from experiments.librispeech import reference_items, cross_session_split
        ref_all = reference_items(items, min_sec=4.0, utt_index=args.ref_utt)
        split = cross_session_split(items, min_sec=4.0)
        cand = sorted(ref_all)
        if args.cross_session:
            cand = [s for s in cand if s in split
                    and len(split[s]["enroll"]) > args.enroll_utt]
        spks = cand[args.spk_offset:args.spk_offset + args.n_files]
        ref_of = {s: ref_all[s] for s in spks}
        if args.cross_session:
            enr_of = {s: split[s]["enroll"][args.enroll_utt] for s in spks}
            print(f"[clone] {len(spks)} speakers, enrolment from a DIFFERENT "
                  f"chapter than the reference (shared selector)")
        else:
            ps_all = per_speaker(items, n_utt=args.enroll_utt + 1)
            enr_of = {s: ps_all[s][args.enroll_utt] for s in spks
                      if len(ps_all.get(s, [])) > args.enroll_utt}
            spks = [s for s in spks if s in enr_of]
            ref_of = {s: ref_of[s] for s in spks}
            print(f"[clone] {len(spks)} speakers, SAME-SESSION enrolment "
                  f"(shared reference selector)")
    else:
        ps = per_speaker(items, n_utt=max(args.ref_utt, args.enroll_utt) + 1)
        spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]
        ref_of = {s: ps[s][args.ref_utt] for s in spks}
        enr_of = {s: ps[s][args.enroll_utt] for s in spks}
        print("[clone] SAME-SESSION (legacy): enrolment may share a chapter with "
              "the reference; see fix S1.")

    rows, t_start = [], time.time()
    for i, spk in enumerate(spks):
        src_spk = spks[(i + 1) % len(spks)]
        it = ref_of[spk]
        name = f"spk{spk}_{it['chapter']}_{it['utt']:04d}"
        wav_o = load_flac(it["path"], 4.0).to(device)
        enr = load_flac(enr_of[spk]["path"], 4.0).to(device)
        src = load_flac(ref_of[src_spk]["path"], 4.0).to(device)
        e_enr = real_embed_all(encs, enr)

        # the reference audio for each arm: unprotected control + each variant
        arms = {"unprotected": wav_o}
        for v in variants:
            p = os.path.join(WAV_DIR, f"{name}_{v}.wav")
            if not os.path.isfile(p):
                print(f"  missing {p}"); continue
            y, _ = sf.read(p, dtype="float32")
            arms[v] = torch.from_numpy(y).reshape(1, -1).to(device)[:, : wav_o.shape[-1]]

        for pname, pur in purs.items():
            for arm, ref_wav in arms.items():
                if pur is None:                                   # phonepure: pre-purified on disk
                    utt = name.replace("spk", "", 1).replace("_", "-")   # spk1089_134686_0000 -> 1089-134686-0000
                    pp = os.path.join(PROJ, "external", "De-AntiFake", "data", f"ours_{arm}",
                                      "audio-purified-PhonePuRe_t3_step1", f"libri_p{utt}.wav")
                    if not os.path.isfile(pp):
                        print(f"  missing PhonePuRe output {pp}"); continue
                    y, _ = sf.read(pp, dtype="float32")
                    ref_pur = torch.from_numpy(y).reshape(1, -1).to(device)[:, : wav_o.shape[-1]]
                else:
                    with torch.no_grad():
                        ref_pur = pur(ref_wav)
                # DiffVC is a DIFFUSION model: generator.forward draws noise for
                # each of its 30 timesteps from the global RNG, so two runs of
                # this script produce different clones and different binary
                # rates (RESULTS 31b measured 32 % of unprotected cells flipping
                # between identical runs). Every cloning CSV in the project was
                # written unseeded, which is why --clone-seed defaults to off:
                # passing it would change v1's numbers.
                #
                # When given, the seed is mixed with (speaker, arm, purifier) so
                # each clone is reproducible on its own, independent of loop
                # order or of which arms a run happened to include.
                if args.clone_seed is not None:
                    h = zlib.crc32(f"{spk}|{arm}|{pname}".encode()) & 0xFFFFFFFF
                    torch.manual_seed((args.clone_seed * 1000003 + h) % (2 ** 63))
                clone = vc.clone(src, ref_pur).to(device)
                # Keep the clone. Council #6: there is no quality evidence on
                # what the ATTACKER actually produces, only on what we ship. A
                # protection that merely makes the clone unusable-sounding is a
                # different (weaker) claim than one that moves the speaker
                # identity, and the two cannot be told apart without these
                # files. ~22 MB for the full run; scored afterwards by
                # eval_quality_nr.py. Off by default so v1 reruns are unchanged.
                if args.save_clones:
                    cdir = os.path.join(WAV_DIR, "clones")
                    os.makedirs(cdir, exist_ok=True)
                    sf.write(os.path.join(
                        cdir, f"{name}__{arm}__{pname}__clone.wav"),
                        clone.detach().cpu().reshape(-1).numpy(), 16000)
                e_cl = real_embed_all(encs, clone)
                for n in e_enr:
                    sim = cs(e_cl[n], e_enr[n])
                    tau = THR[n]
                    rows.append(dict(speaker=spk, source_speaker=src_spk, variant=arm,
                                     purifier=pname, encoder=n,
                                     clone_sim=round(sim, 4), threshold=round(tau, 4),
                                     margin=round(sim - tau, 4),
                                     clone_succeeds=bool(sim > tau),
                                     # ECAPA only: the field convention, for comparability
                                     succeeds_at_025=bool(sim > FIELD_CONVENTION_THRESHOLD)
                                     if n == "ECAPA-TDNN" else None,
                                     **PROV))
        el = time.time() - t_start
        print(f"  [{i+1}/{len(spks)}] spk{spk}  ({el/60:.1f} min elapsed, "
              f"~{el/(i+1)*(len(spks)-i-1)/60:.0f} min left)", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)   # checkpoint each speaker

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # ---- AntiFake pre-filter, PER (ENCODER, PURIFIER).
    #
    # Fix #4 (council 2026-10-04, PLAN_V3 W1). This block used to key the
    # keep-set on the ENCODER alone, which unions across purifier conditions: a
    # speaker who cloned only with NO purifier was then scored under `ours` and
    # `diffwave` too, where the unprotected attack had already failed. Those are
    # free protection successes, and excluding them is the entire purpose of
    # AntiFake's filter ("given that all samples have already been filtered to
    # bypass SV without AntiFake").
    #
    # Measured on cloning_tc_m30_seg_s7.csv, the union made protection look
    # better everywhere it mattered, worst on ECAPA under the foreign purifier:
    #   ECAPA / diffwave   correct 33.3 % (n=6)    union 11.8 % (n=17)
    #   WavLM / diffwave   correct 23.1 % (n=13)   union 22.2 % (n=18)
    #   d-vec / diffwave   correct 60.9 % (n=23)   union 57.7 % (n=26)
    # (clone-success rate on the protected arm; lower looks better for us.)
    #
    # The correct implementation already existed in analyse_cloning.prefilter
    # and the two drifted apart. There is now ONE definition, imported, so they
    # cannot drift again.
    #
    # NOTE (PLAN_V3 2a): the filtered BINARY rate printed below is no longer the
    # paper's cloning headline. 32 % of unprotected cells flip between identical
    # runs, so the keep-set is itself a noisy draw of the defender's own outcome,
    # and the n=6 cell above is why. The headline is the UNFILTERED paired
    # continuous margin over all speakers: experiments/analyse_clone_margin.py.
    # These rates are kept because prior work reports them.
    from experiments.analyse_cloning import prefilter
    f = prefilter(df)
    print("\n" + "=" * 78)
    print("AntiFake pre-filter -- keep only speakers whose UNPROTECTED audio cloned")
    print("keyed per (encoder, purifier); see fix #4 in the source")
    for n in sorted(df.encoder.unique()):
        for pur in sorted(df.purifier.unique()):
            ch = f[(f.encoder == n) & (f.purifier == pur)]
            print(f"    {n:<22} {pur:<10} {ch.speaker.nunique()}/"
                  f"{df.speaker.nunique()} speakers retained")
    print("=" * 78)
    if not len(f):
        raise SystemExit("pre-filter removed everything: the attacker never cloned "
                         "clean audio at these thresholds. Fix the attack first.")

    for pname in pur_names:
        s = f[f.purifier == pname]
        if not len(s):
            continue
        print(f"\n--- attacker purifier: {pname} ---")
        piv = s.pivot_table(index="variant", columns="encoder", values="clone_sim",
                            aggfunc="mean").round(3)
        rate = (100 * s.pivot_table(index="variant", columns="encoder",
                                    values="clone_succeeds", aggfunc="mean")).round(0)
        order = [v for v in ["unprotected"] + variants if v in piv.index]
        print("mean clone similarity (lower = better protection):")
        print(piv.loc[order].to_string())
        print("\nclone success rate %, i.e. the attack WORKED (lower = better protection):")
        print(rate.loc[order].to_string())
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
