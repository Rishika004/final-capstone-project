"""
experiments/prep_f5_prompts.py
==============================
N2 stage A -- build everything the isolated F5-TTS venv needs, and nothing else.

PREREG: paper/PREREG_N2_modern_cloner.md

Why three stages. F5-TTS lives in its own venv (external/f5tts_venv) because it
pins transformers/torch ranges the project cannot take. That venv has none of
this project's code, so the purifier cannot run inside it. This stage therefore
does every project-side thing up front:

  1. select the SAME 29 speakers, the SAME reference and enrolment utterances
     as exp_cloning_protection.py (shared selector, --cross-session);
  2. transcribe each victim's CLEAN 4-s crop with Whisper -> ref_text, and the
     SOURCE speaker's clean crop -> gen_text (prereg 2.4: the attacker is given
     the true words, which is the attacker-favourable choice);
  3. write every (arm x channel) reference wav to results/f5_refs/ with the
     attacker's purifier already applied.

Stage B (experiments/f5_clone.py, run by the f5 venv) then needs only wavs and
a CSV. Stage C (experiments/score_f5_clones.py) scores the clones with the
project's encoders and thresholds.

Run:
  python experiments/prep_f5_prompts.py --prereg paper/PREREG_N2_modern_cloner.md
"""
import sys, io, os, argparse, time

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
import torch._dynamo   # noqa: F401  (must precede speechbrain; see memory)
import transformers    # noqa: F401
import soundfile as sf
import pandas as pd

from experiments.exp_purifier_aware import WAV_DIR
from experiments.librispeech import list_librispeech, load_flac, EVAL_SUBSET
from experiments.exp_cloning_protection import build_purifiers
from utils.provenance import provenance, add_prereg_arg

SR = 16000
REF_DIR = os.path.join(PROJ, "experiments", "results", "f5_refs")
OUT = os.path.join(PROJ, "experiments", "results", "f5_prompts.csv")

# The three arms of prereg 2.1. unprotected is the GATE (prereg 4).
ARMS = ["unprotected", "plain_tc_m30_seg", "fullnsT_tc_m30_seg"]


def transcript_of(item, crop_sec=4.0):
    """LibriSpeech's own transcript for the FULL utterance.

    Returned for reference only -- it is NOT used as ref_text, because the
    scored audio is the first `crop_sec` seconds and the transcript covers the
    whole utterance. Using it would tell F5-TTS the clip says words that are
    not in it. Whisper on the actual crop is what we use instead.
    """
    d = os.path.dirname(item["path"])
    base = os.path.basename(item["path"]).rsplit(".", 1)[0]
    trans = os.path.join(d, f"{item['speaker']}-{item['chapter']}.trans.txt")
    if not os.path.isfile(trans):
        return ""
    with open(trans, "r", encoding="utf-8") as fh:
        for line in fh:
            k, _, text = line.partition(" ")
            if k == base:
                return text.strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=0, dest="spk_offset")
    ap.add_argument("--ref-utt", type=int, default=0, dest="ref_utt")
    ap.add_argument("--enroll-utt", type=int, default=0, dest="enroll_utt")
    ap.add_argument("--subset", default=EVAL_SUBSET)
    ap.add_argument("--purifiers", default="none,ours,diffwave")
    ap.add_argument("--dw-t", type=int, default=3, dest="dw_t")
    ap.add_argument("--ours-ckpt",
                    default=os.path.join(PROJ, "checkpoints", "purifier_v7.pt"),
                    dest="ours_ckpt")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--asr-model", default="openai/whisper-small", dest="asr_model")
    ap.add_argument("--out", default=OUT)
    add_prereg_arg(ap)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROV = provenance(args.prereg)   # code_sha / prereg_sha / argv per row (W0)
    arms = [a for a in args.arms.split(",") if a]
    os.makedirs(REF_DIR, exist_ok=True)

    # ---- the SAME selection as exp_cloning_protection.py -------------------
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             args.subset, 4.0)
    from experiments.librispeech import reference_items, cross_session_split
    ref_all = reference_items(items, min_sec=4.0, utt_index=args.ref_utt)
    split = cross_session_split(items, min_sec=4.0)
    cand = sorted(s for s in sorted(ref_all)
                  if s in split and len(split[s]["enroll"]) > args.enroll_utt)
    spks = cand[args.spk_offset:args.spk_offset + args.n_files]
    ref_of = {s: ref_all[s] for s in spks}
    enr_of = {s: split[s]["enroll"][args.enroll_utt] for s in spks}
    print(f"[prep] {len(spks)} speakers, cross-session enrolment: {spks}")
    if len(spks) != args.n_files:
        print(f"[prep] WARNING: wanted {args.n_files}, got {len(spks)}")

    # ---- Whisper on the CLEAN crops ---------------------------------------
    # Loaded the same way eval_quality_nr.py loads it (the ASR pipeline needs
    # torchcodec/FFmpeg, which this machine does not have).
    from experiments.eval_quality_nr import load_asr
    print(f"[prep] loading ASR {args.asr_model}")
    transcribe = load_asr(device, args.asr_model)

    text_of, wav_o_of = {}, {}
    for spk in spks:
        wav_o = load_flac(ref_of[spk]["path"], 4.0)
        wav_o_of[spk] = wav_o
        t = transcribe(wav_o.squeeze(0).numpy()).strip()
        text_of[spk] = t
        print(f"  spk{spk}: {t!r}")
    del transcribe
    torch.cuda.empty_cache()

    # ---- purifiers ---------------------------------------------------------
    pur_names = [p for p in args.purifiers.split(",") if p]
    purs = build_purifiers(pur_names, device, dw_t=args.dw_t,
                           ours_ckpt=args.ours_ckpt)
    print(f"[prep] purifiers: {list(purs)}")

    rows, t0, n_written, n_missing = [], time.time(), 0, 0
    for i, spk in enumerate(spks):
        src_spk = spks[(i + 1) % len(spks)]        # same rotation as the VC run
        it = ref_of[spk]
        name = f"spk{spk}_{it['chapter']}_{it['utt']:04d}"
        wav_o = wav_o_of[spk].to(device)

        refs = {"unprotected": wav_o}
        for v in arms:
            if v == "unprotected":
                continue
            p = os.path.join(WAV_DIR, f"{name}_{v}.wav")
            if not os.path.isfile(p):
                print(f"  MISSING crafted wav {p}")
                n_missing += 1
                continue
            y, _ = sf.read(p, dtype="float32")
            refs[v] = torch.from_numpy(y).reshape(1, -1).to(device)[:, : wav_o.shape[-1]]

        for pname, pur in purs.items():
            for arm, ref_wav in refs.items():
                if pur is None:
                    raise SystemExit("[prep] phonepure is not part of N2 (prereg 2.2)")
                with torch.no_grad():
                    ref_pur = pur(ref_wav)
                rp = os.path.join(REF_DIR, f"{name}__{arm}__{pname}.wav")
                sf.write(rp, ref_pur.detach().cpu().reshape(-1).numpy(), SR)
                n_written += 1
                rows.append(dict(
                    speaker=spk, source_speaker=src_spk, name=name,
                    variant=arm, purifier=pname,
                    ref_wav=os.path.relpath(rp, PROJ).replace("\\", "/"),
                    ref_text=text_of[spk],
                    gen_text=text_of[src_spk],
                    enrol_path=os.path.relpath(enr_of[spk]["path"], PROJ).replace("\\", "/"),
                    ref_path=os.path.relpath(it["path"], PROJ).replace("\\", "/"),
                    libri_transcript_full=transcript_of(it),
                    **PROV))
        print(f"[prep] {i+1}/{len(spks)} spk{spk} ({time.time()-t0:.0f} s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[prep] {len(df)} rows -> {args.out}")
    print(f"[prep] {n_written} reference wavs -> {REF_DIR}")
    if n_missing:
        print(f"[prep] {n_missing} crafted wavs MISSING -- those cells are absent")
    print("[prep] arms x channels present:")
    print(df.groupby(["variant", "purifier"]).size().to_string())


if __name__ == "__main__":
    main()
