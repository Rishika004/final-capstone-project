"""
experiments/prep_phonepure_inputs.py
====================================
Build the input folder De-AntiFake's PhonePuRe stage 2 expects, for ONE arm.

WHAT THEIR CODE NEEDS (external/De-AntiFake/PhonePuRe/refinement_models/phoneme_utils.py)
  audio_dir/          libri_p{spk}-{chap}-{utt}.wav      16 kHz mono
  text_dir/           libri_p{spk}-{chap}-{utt}.txt      transcript (kept for completeness)
  audio_dir-textgrid/metadata.txt                        one line per wav:
        stamps|durations|phones|{spk}-{chap}-{utt}.wav|{spk}
  If metadata.txt exists, create_metadata_from_audio_text_mfa() returns it and
  NEVER calls MFA (line 238). That is how the aligner is bypassed on Windows.

WHERE THE PHONEMES COME FROM
  gilkeyio/librispeech-alignments (HF, CC-BY-4.0): MFA english_us_arpa alignments
  of the ORIGINAL utterances. Our perturbation is additive and time-preserving and
  our crop is the FIRST 4.0 s (librispeech.load_flac), so the original timings
  apply unchanged, clipped at 4.0 s. Frame = floor(t * 16000 / 128), matching
  their time2frame; gaps and 'spn' become 'sil', exactly as their converter does.

Run:
  python experiments/prep_phonepure_inputs.py --tag fullnsT_tc_m30_seg
  python experiments/prep_phonepure_inputs.py --tag unprotected
Output:
  external/De-AntiFake/data/ours_{tag}/{audio,text}/ and ours_{tag}/audio-textgrid/metadata.txt
"""
import os
import sys
import math
import argparse

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
import soundfile as sf

SR, HOP, CROP_SEC = 16000, 128, 4.0
WAV_DIR = os.path.join(PROJ, "experiments", "results", "purifier_aware")
DAF = os.path.join(PROJ, "external", "De-AntiFake")
PARQUET = os.path.join(DAF, "data", "librispeech_alignments",
                       "test_clean_phoneme_alignments.parquet")
SIL = {"", "sp", "spn", "sil"}


def t2f(t):
    return math.floor((t + 1e-9) * SR / HOP)


def metadata_line(phonemes, utt_id, crop=CROP_SEC):
    """Their converter's logic, applied to a list of {phoneme,start,end} dicts
    and clipped to the crop. Returns 'stamps|durations|phones|basename|speaker'."""
    phones, stamps, durs = [], [], []
    prev = 0.0
    for ph in phonemes:
        s, e, p = float(ph["start"]), float(ph["end"]), ph["phoneme"]
        if s >= crop:
            break
        e = min(e, crop)
        if s > prev:                                   # gap -> silence
            phones.append("sil"); stamps.append(t2f(prev)); durs.append(t2f(s) - t2f(prev))
        phones.append("sil" if p in SIL else p)
        sf_, ef_ = t2f(s), max(t2f(e), t2f(s) + 1)
        stamps.append(sf_); durs.append(ef_ - sf_)
        prev = e
    if prev < crop:                                    # trailing silence to crop end
        phones.append("sil"); stamps.append(t2f(prev)); durs.append(t2f(crop) - t2f(prev))
    spk = utt_id.split("-")[0]
    return "|".join([" ".join(map(str, stamps)), " ".join(map(str, durs)),
                     " ".join(phones), utt_id + ".wav", spk])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="a crafted tag (files spk*_{tag}.wav in purifier_aware/) or "
                         "'unprotected' for the clean 4-s crops of the same utterances")
    ap.add_argument("--ref-tag", default="fullnsT_tc_m30_seg", dest="ref_tag",
                    help="which tag's file list defines the 29 utterances when "
                         "--tag unprotected")
    args = ap.parse_args()
    from experiments.librispeech import load_flac

    src_tag = args.ref_tag if args.tag == "unprotected" else args.tag
    files = sorted(f for f in os.listdir(WAV_DIR) if f.endswith(f"_{src_tag}.wav"))
    if not files:
        sys.exit(f"no files *_{src_tag}.wav in {WAV_DIR}")
    align = pd.read_parquet(PARQUET).set_index("id")

    out = os.path.join(DAF, "data", f"ours_{args.tag}")
    adir, tdir, gdir = (os.path.join(out, "audio"), os.path.join(out, "text"),
                        os.path.join(out, "audio-textgrid"))
    for d in (adir, tdir, gdir):
        os.makedirs(d, exist_ok=True)

    lines, missing = [], []
    for f in files:
        stem = f[: -len(f"_{src_tag}.wav")]              # spk1089_134686_0000
        _, spk, chap, utt = stem.replace("spk", "spk_", 1).split("_")
        utt_id = f"{spk}-{chap}-{utt}"                    # 1089-134686-0000
        if utt_id not in align.index:
            missing.append(utt_id); continue
        name = f"libri_p{utt_id}"
        if args.tag == "unprotected":
            flac = os.path.join(PROJ, "data", "librispeech", "LibriSpeech", "test-clean",
                                spk, chap, utt_id + ".flac")
            wav = load_flac(flac, max_sec=CROP_SEC).squeeze(0).numpy()
        else:
            wav, sr = sf.read(os.path.join(WAV_DIR, f), dtype="float32")
            assert sr == SR, (f, sr)
            wav = wav[: int(CROP_SEC * SR)]
        sf.write(os.path.join(adir, name + ".wav"), wav.astype(np.float32), SR)
        row = align.loc[utt_id]
        with open(os.path.join(tdir, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(str(row["transcript"]).strip() + "\n")
        lines.append(metadata_line(list(row["phonemes"]), utt_id))

    with open(os.path.join(gdir, "metadata.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[prep] tag={args.tag}: {len(lines)} utterances -> {out}")
    print(f"[prep] metadata: {os.path.join(gdir, 'metadata.txt')}")
    if missing:
        print(f"[prep] WARNING no alignment for: {missing}")
    print("[prep] example line:", lines[0][:160], "...")


if __name__ == "__main__":
    main()
