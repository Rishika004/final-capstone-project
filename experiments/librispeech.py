"""
experiments/librispeech.py
==========================
Minimal LibriSpeech reader using soundfile (torchaudio 2.9's dataset class
routes through torchcodec/FFmpeg, which is not available on this machine).

    items = list_librispeech(root, EVAL_SUBSET, min_sec=4.0)
    wav   = load_flac(path, max_sec=4.0)      # [1, T] float32 16 kHz, peak 0.9

SPLIT POLICY (changed 2026-09-17; see paper/FIXES_TO_PUBLISHABLE.md S1b)
------------------------------------------------------------------------
Everything used to run on `dev-clean` -- purifier training AND evaluation, on two
halves of the same 40-speaker split. That was wrong three ways:

  * **Zero of seven baseline papers evaluate on dev-clean.** AntiFake and
    De-AntiFake both name `test-clean` explicitly, so our numbers were not
    comparable to the two papers the work is positioned against.
  * **n = 20 was a ceiling, not a choice** -- dev-clean has exactly 40 speakers and
    the purifier consumed half of them.
  * **Same-split train/eval.** De-AntiFake trains its purifier on `train-clean-100`
    and states the disjointness; we did not.

It also caused the implausible ECAPA EER of 0.03 %: every dev-clean speaker's first
four utterances come from ONE chapter, so thresholds were calibrated on
same-session trials. VocalBridge, with the same limitation, reports 0.6 % -- twenty
times higher.
"""
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 16000

# ---------------------------------------------------------------------------
# Splits. Import these rather than hardcoding a subset name.
# ---------------------------------------------------------------------------
EVAL_SUBSET = "test-clean"        # what AntiFake and De-AntiFake report on
TRAIN_SUBSET = "train-clean-100"  # purifier training, disjoint from evaluation

# De-AntiFake's published evaluation set (github.com/cyberrrange/De-AntiFake,
# data/test_set_example_clean/audio/ -- 25 speakers x 5 utterances, 9M/16F).
# Using these exact speakers gives one table row directly comparable to their
# Table 1 and to AntiFake's Table 3.
DEANTIFAKE_SPEAKERS = [
    1089, 1188, 1221, 1284, 1320, 1580, 1995, 2094, 2300, 2830, 2961, 3570,
    3575, 3729, 4077, 4446, 4507, 4970, 4992, 5105, 5142, 5639, 5683, 6829, 6930,
]

# test-clean speakers with >= 2 chapters -- the ones where enrolment can come from
# a DIFFERENT recording session than the reference. The other 11 have a single
# chapter and cannot support cross-session calibration.
CROSS_SESSION_SPEAKERS = [
    61, 121, 237, 260, 908, 1089, 1221, 1284, 1320, 1580, 1995, 2830, 2961,
    3570, 4077, 4446, 4970, 4992, 5105, 5142, 5683, 6829, 6930, 7021, 7127,
    7176, 8224, 8463, 8555,
]


def chapters_of(items):
    """speaker -> sorted list of chapter ids present in `items`."""
    out = {}
    for it in items:
        out.setdefault(it["speaker"], set()).add(it["chapter"])
    return {k: sorted(v) for k, v in out.items()}


def cross_session_split(items, min_sec=4.0):
    """Per speaker, split utterances into (reference_chapter, enrolment_chapter).

    Reference comes from the speaker's LARGEST chapter, enrolment from a different
    one. Speakers with a single chapter are dropped -- scoring a trial whose target
    and enrolment share a chapter is what produced our 0.03 % EER.

    Returns {speaker: {"ref": [items], "enroll": [items]}}.
    """
    by_spk = {}
    for it in items:
        if it["seconds"] >= min_sec:
            by_spk.setdefault(it["speaker"], []).append(it)
    out = {}
    for spk, its in by_spk.items():
        by_ch = {}
        for it in its:
            by_ch.setdefault(it["chapter"], []).append(it)
        if len(by_ch) < 2:
            continue                      # cannot do cross-session for this speaker
        order = sorted(by_ch, key=lambda c: -len(by_ch[c]))
        out[spk] = {"ref": by_ch[order[0]], "enroll": by_ch[order[1]]}
    return out


def reference_items(items, min_sec=4.0, speakers=None, utt_index=0):
    """The protected REFERENCE utterance per speaker, chosen identically everywhere.

    ONE function, called by crafting AND by the cloning evaluation, because having
    two code paths agree by coincidence is not a guarantee. It stopped being one on
    2026-09-19: crafting selected each speaker's first utterance overall while the
    cloning step asked for the first utterance of the speaker's largest chapter.
    Those differ whenever a speaker's opening file is not in their biggest chapter,
    which was 9 of 18 speakers. The clone step would have silently skipped them.

    Rule: prefer the LARGEST chapter, so a cross-session enrolment is available for
    the same speaker; fall back to the first utterance for single-chapter speakers,
    so they can still be crafted and scored under a same-session protocol.

    Returns {speaker: item}. Threshold protocol is a SEPARATE choice -- this is only
    about which file is protected.
    """
    by_spk = {}
    for it in items:
        if it["seconds"] >= min_sec and (speakers is None or it["speaker"] in speakers):
            by_spk.setdefault(it["speaker"], []).append(it)
    out = {}
    for spk, its in by_spk.items():
        by_ch = {}
        for it in its:
            by_ch.setdefault(it["chapter"], []).append(it)
        order = sorted(by_ch, key=lambda c: (-len(by_ch[c]), c))
        lst = by_ch[order[0]]
        if len(lst) > utt_index:
            out[spk] = lst[utt_index]
    return out


def list_librispeech(root, subset="dev-clean", min_sec=0.0):
    """Return sorted list of dicts: speaker, chapter, utt, path, seconds."""
    base = Path(root) / "LibriSpeech" / subset
    if not base.is_dir():
        raise FileNotFoundError(f"LibriSpeech subset not found: {base}")
    items = []
    for flac in sorted(base.rglob("*.flac")):
        info = sf.info(str(flac))
        sec = info.frames / info.samplerate
        if sec < min_sec:
            continue
        spk, ch, utt = flac.stem.split("-")
        items.append(dict(speaker=int(spk), chapter=int(ch), utt=int(utt),
                          path=str(flac), seconds=sec, sr=info.samplerate))
    return items


def load_flac(path, max_sec=None):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)             # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        import torchaudio  # only reached for non-16 kHz sources
        wav = torchaudio.functional.resample(wav, sr, SR)
    if max_sec is not None:
        wav = wav[:, : int(max_sec * SR)]
    return wav / (wav.abs().max() + 1e-8) * 0.9


def per_speaker(items, n_utt, n_spk=None):
    """First n_utt items per speaker, first n_spk speakers (sorted by id)."""
    by = {}
    for it in items:
        lst = by.setdefault(it["speaker"], [])
        if len(lst) < n_utt:
            lst.append(it)
    spks = sorted(by)
    if n_spk is not None:
        spks = spks[:n_spk]
    return {s: by[s] for s in spks}
