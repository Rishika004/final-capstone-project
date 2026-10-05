"""
experiments/eval_quality_nr.py
==============================
Fair perceptual measurement for protections that CHANGE the voice.

PESQ/STOI are reference-based: they score any deviation from the original as
damage, so they punish a formant warp that a listener would accept. Our warp
settings score PESQ 1.2 while protecting 95 % of speakers -- that number cannot
decide anything. This script measures the two things that can be measured
without a reference, plus the same numbers for the competing additive methods:

  naturalness    UTMOS (no-reference MOS predictor) and torchaudio SQUIM
                 (reference-free STOI / PESQ / SI-SDR estimates).
                 "Does this sound like natural speech?" -- NOT "is it identical
                 to the original?"
  content        Word error rate of Whisper on the protected audio, using
                 Whisper's transcript of the ORIGINAL 4-s crop as reference.
                 (Ground-truth LibriSpeech transcripts cover the whole
                 utterance, not our crop, so the original's own transcript is
                 the correct reference for "did the transform damage speech?")

What this CANNOT measure, and must go to a listening test: whether the
protected audio still sounds like the SAME PERSON. That matters because a
protection is only usable if the owner can still publish it as their own voice.
A formant warp large enough to defeat speaker verification may also change
perceived identity -- an automated speaker encoder cannot answer this, since
disagreeing with the encoder is the entire point of the method.

Run:
  python experiments/eval_quality_nr.py --n-files 20
Output:
  experiments/results/quality_nr.csv
"""
import sys, io, os, argparse, warnings
# TensorFlow + Keras 3 is installed here; transformers tries the TF Whisper and
# dies. Force the PyTorch path BEFORE transformers is imported anywhere.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
if not getattr(sys.stdout, "_nvp_utf8", False):   # guard: re-wrapping a second
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # time closes the first wrapper
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import soundfile as sf

from evaluation.evaluate import SAMPLE_RATE
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET
from experiments.exp_parametric_probe import TRANSFORMS
from experiments.exp_purifier_aware import WAV_DIR as ADD_WAV_DIR

OUT = os.path.join(PROJ, "experiments", "results", "quality_nr.csv")


# ---------------------------------------------------------------------------
def load_utmos(device):
    """UTMOS22 via torch.hub. torch 2.9's _validate_not_a_forked_repo raises
    KeyError('Authorization') on unauthenticated requests, so bypass that
    check -- it only guards against loading forks, not a safety mechanism."""
    import torch.hub as hub
    orig = getattr(hub, "_validate_not_a_forked_repo", None)
    hub._validate_not_a_forked_repo = lambda *a, **k: True
    try:
        m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True, verbose=False)
        return m.to(device).eval()
    finally:
        if orig is not None:
            hub._validate_not_a_forked_repo = orig


def load_squim(device):
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    return SQUIM_OBJECTIVE.get_model().to(device).eval()


def load_asr(device, name):
    """Load Whisper directly rather than via transformers' ASR pipeline: the
    pipeline installs an audio decoder backed by torchcodec, which needs FFmpeg
    DLLs that are not present on this machine. We already hold waveforms in
    memory, so no decoding is needed."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def transcribe(wav_np):
        feats = proc(wav_np, sampling_rate=SAMPLE_RATE,
                     return_tensors="pt").input_features.to(device)
        ids = model.generate(feats, max_new_tokens=128)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]

    return transcribe


def norm_text(s):
    import re
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=20, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--transforms", default="warp,reverb,eq,tilt")
    ap.add_argument("--additive-tags", default="plain_m30_f15,full_m30_f15",
                    dest="additive_tags")
    ap.add_argument("--asr", default="openai/whisper-small.en")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = torch.hann_window(1024, device=device)

    print("[Q] loading UTMOS ...", flush=True)
    utmos = load_utmos(device)
    print("[Q] loading SQUIM ...", flush=True)
    try:
        squim = load_squim(device)
    except Exception as e:
        print(f"[Q] SQUIM unavailable ({e}); continuing without it")
        squim = None
    print(f"[Q] loading ASR {args.asr} ...", flush=True)
    asr = load_asr(device, args.asr)
    import jiwer

    ps = per_speaker(list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                                      EVAL_SUBSET, 4.0), n_utt=args.utt_index + 1)
    spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]
    tnames = [t for t in args.transforms.split(",") if t in TRANSFORMS]
    tags = [t for t in args.additive_tags.split(",") if t]

    def measure(wav, ref_words=None):
        w = wav.reshape(1, -1).float().to(device)
        with torch.no_grad():
            mos = float(utmos(w, SAMPLE_RATE))
            sq = {}
            if squim is not None:
                st, pe, si = squim(w)
                sq = dict(squim_stoi=float(st), squim_pesq=float(pe),
                          squim_sisdr=float(si))
        txt = asr(w.squeeze(0).cpu().numpy()).strip()
        words = norm_text(txt)
        wer = (jiwer.wer(" ".join(ref_words), " ".join(words))
               if ref_words is not None and ref_words and words else float("nan"))
        return dict(utmos=round(mos, 3), wer=round(wer, 4) if wer == wer else np.nan,
                    n_words=len(words), **{k: round(v, 3) for k, v in sq.items()}), words

    rows = []
    for si, spk in enumerate(spks):
        it = ps[spk][args.utt_index]
        name = f"spk{spk}_{it['chapter']}_{it['utt']:04d}"
        wav_o = load_flac(it["path"], 4.0).to(device)
        m_o, ref_words = measure(wav_o)
        rows.append(dict(speaker=spk, family="original", condition="original",
                         magnitude=np.nan, wer=0.0, **{k: v for k, v in m_o.items() if k != "wer"}))

        for tname in tnames:
            fn, mags, unit = TRANSFORMS[tname]
            for mag in mags:
                try:
                    with torch.no_grad():
                        wav_p = fn(wav_o, mag, win, seed=spk).clamp(-1, 1)[:, : wav_o.shape[-1]]
                except Exception as exc:
                    print(f"  {name} {tname} {mag}: FAILED ({exc})"); continue
                m, _ = measure(wav_p, ref_words)
                rows.append(dict(speaker=spk, family="on-manifold",
                                 condition=f"{tname} {mag}", magnitude=mag, **m))

        for tag in tags:
            p = os.path.join(ADD_WAV_DIR, f"{name}_{tag}.wav")
            if not os.path.isfile(p):
                continue
            y, _ = sf.read(p, dtype="float32")
            wav_p = torch.from_numpy(y).reshape(1, -1).to(device)[:, : wav_o.shape[-1]]
            m, _ = measure(wav_p, ref_words)
            rows.append(dict(speaker=spk, family="additive", condition=tag,
                             magnitude=np.nan, **m))
        print(f"[Q] {name} done ({si+1}/{len(spks)})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    cols = [c for c in ["utmos", "wer", "squim_stoi", "squim_pesq", "squim_sisdr"] if c in df]
    summ = df.groupby(["family", "condition"], sort=False)[cols].mean().round(3)
    print("\n" + "=" * 84)
    print("NO-REFERENCE QUALITY  (UTMOS 1-5, higher better; WER vs the original's "
          "own transcript)")
    print("=" * 84)
    print(summ.to_string())
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
