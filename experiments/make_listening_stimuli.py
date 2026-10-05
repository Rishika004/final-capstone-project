"""experiments/make_listening_stimuli.py -- blinded, loudness-matched stimuli.

Rebuilds the listening-test audio from the SHIPPING arm, fixing all four defects
council #6 found in `listening_test/audio`:

  1. **Condition visible in the filename.** The old files are named
     `3000_additive_eot.wav`, `3000_original.wav`. Anyone inspecting the folder --
     including a listener who looks -- sees the condition. Here every stimulus is
     `stim_<16 hex>.wav`, where the hex is HMAC(key, speaker|condition) and the
     key is written to a separate file that does not ship with the audio.
  2. **Stale arms.** The old conditions point at `plain_m30_f15` / `full_m30_f15`,
     dev-clean arm names from the old base, and at `full` rather than the
     shipping `fullnsT`. Here the arms are the corrected-base
     `*_tc_m30_seg` set.
  3. **No loudness normalisation.** Protected files are quieter than originals
     by construction (SNR-constrained perturbation), so level was confounded
     with condition -- the one confound a MOS test must not have. Everything is
     normalised to -23 LUFS with `utils/loudness.py` (BS.1770-4), and any file
     whose peak ceiling clamped before reaching the target is reported.
  4. **Wrong speaker selector.** The old script used `per_speaker`, picking a
     different speaker set and utterance than the runs it describes. Here the
     speakers come from the crafted wav filenames themselves, so the stimuli are
     by construction the files the paper reports.

Outputs, under `listening_test/`:
  stimuli/stim_<hash>.wav   the blinded audio
  stimuli_key.csv           hash -> speaker, condition, source file, loudness
  stimuli_manifest.json     the build record (arms, target, counts, warnings)

Run:
  python experiments/make_listening_stimuli.py --n-speakers 8
"""
import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from utils.loudness import integrated_lufs, normalise

WAV_DIR = os.path.join(PROJ, "experiments", "results", "purifier_aware")
LIBRI = os.path.join(PROJ, "data", "librispeech", "LibriSpeech")
OUT_DIR = os.path.join(PROJ, "listening_test")
SR = 16000

# Corrected-base shipping arms. `original` and `anchor_noise` are synthesised
# from the same source utterance, so every condition for a speaker is the same
# words in the same voice and only the processing differs.
CONDITIONS = [
    ("original",   None,                 "high anchor: must be rated >= 4"),
    ("anchor_noise", None,               "low anchor, 0 dB SNR: must be rated <= 2"),
    ("plain",      "plain_tc_m30_seg",   "PGD, purifier not in the loop"),
    ("nsT",        "nsT_tc_m30_seg",     "constraint without EOT"),
    ("full",       "full_tc_m30_seg",    "EOT, purifier-aware"),
    ("fullnsT",    "fullnsT_tc_m30_seg", "EOT + constraint (the shipping arm)"),
]


def blind_name(key, speaker, condition):
    """Stable, unguessable stimulus id. HMAC so the mapping cannot be
    reconstructed from the id alone even though it is deterministic."""
    msg = f"{speaker}|{condition}".encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()[:16]


def add_noise(x, snr_db=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(x.shape)
    sx, sn = float(np.mean(x ** 2)), float(np.mean(n ** 2))
    if sn <= 0 or sx <= 0:
        return x
    return x + n * np.sqrt(sx / sn / (10 ** (snr_db / 10.0)))


def original_path(stem):
    """spk1089_134686_0000 -> <LIBRI>/test-clean/1089/134686/1089-134686-0000.flac"""
    spk, chap, utt = stem.replace("spk", "").split("_")
    for subset in ("test-clean", "dev-clean"):
        p = os.path.join(LIBRI, subset, spk, chap, f"{spk}-{chap}-{utt}.flac")
        if os.path.isfile(p):
            return p, subset
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-speakers", type=int, default=8, dest="n_speakers")
    ap.add_argument("--target-lufs", type=float, default=-23.0, dest="target")
    ap.add_argument("--max-sec", type=float, default=4.0, dest="max_sec")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    ref_arm = "fullnsT_tc_m30_seg"
    stems = sorted(
        f[: -len(f"_{ref_arm}.wav")]
        for f in os.listdir(WAV_DIR)
        if f.endswith(f"_{ref_arm}.wav")
    )
    if not stems:
        raise SystemExit(f"no {ref_arm} wavs in {WAV_DIR}")

    # Every condition must exist for a speaker or that speaker is dropped: a
    # speaker present in some conditions and absent from others would make the
    # per-condition means incomparable.
    arms = [arm for _, arm, _ in CONDITIONS if arm]
    complete = []
    for stem in stems:
        if all(os.path.isfile(os.path.join(WAV_DIR, f"{stem}_{arm}.wav"))
               for arm in arms) and original_path(stem)[0]:
            complete.append(stem)
    print(f"[stim] {len(stems)} speakers with {ref_arm}, "
          f"{len(complete)} with every condition and an original")

    rng = np.random.default_rng(a.seed)
    chosen = sorted(rng.choice(complete, size=min(a.n_speakers, len(complete)),
                               replace=False).tolist())
    print(f"[stim] selected {len(chosen)}: {', '.join(s.split('_')[0] for s in chosen)}")

    key = secrets.token_bytes(32)
    stim_dir = os.path.join(a.out, "stimuli")
    os.makedirs(stim_dir, exist_ok=True)

    rows, warnings = [], []
    for stem in chosen:
        src_flac, subset = original_path(stem)
        orig, sr = sf.read(src_flac, dtype="float64", always_2d=False)
        if sr != SR:
            warnings.append(f"{stem}: original is {sr} Hz, expected {SR}")
        orig = orig[: int(a.max_sec * sr)]

        for cond, arm, _note in CONDITIONS:
            if cond == "original":
                x, src = orig, os.path.relpath(src_flac, PROJ)
            elif cond == "anchor_noise":
                x = add_noise(orig, 0.0, seed=abs(hash(stem)) % (2 ** 31))
                src = os.path.relpath(src_flac, PROJ) + " +0dB noise"
            else:
                p = os.path.join(WAV_DIR, f"{stem}_{arm}.wav")
                x, sr2 = sf.read(p, dtype="float64", always_2d=False)
                if sr2 != SR:
                    warnings.append(f"{stem}/{cond}: {sr2} Hz")
                src = os.path.relpath(p, PROJ)

            y, info = normalise(x, SR, target_lufs=a.target)
            if info["clamped"]:
                warnings.append(
                    f"{stem}/{cond}: peak clamped, reached "
                    f"{info['lufs_after']} LUFS not {a.target}")
            h = blind_name(key, stem, cond)
            sf.write(os.path.join(stim_dir, f"stim_{h}.wav"), y, SR)
            rows.append(dict(stim_id=h, speaker=stem.split("_")[0].replace("spk", ""),
                             utterance=stem, condition=cond, arm=(arm or ""),
                             source=src, lufs_before=info["lufs_before"],
                             gain_db=info["gain_db"], lufs_after=info["lufs_after"],
                             peak=info["peak"], clamped=info["clamped"]))
        print(f"  {stem}: {len(CONDITIONS)} stimuli", flush=True)

    import pandas as pd
    key_csv = os.path.join(a.out, "stimuli_key.csv")
    pd.DataFrame(rows).to_csv(key_csv, index=False)
    with open(os.path.join(a.out, "stimuli_blind_key.txt"), "w") as fh:
        fh.write(key.hex() + "\n")

    # Verification, printed because a silent normaliser is worth nothing.
    df = pd.DataFrame(rows)
    ok = df[~df.clamped]
    print("\n" + "=" * 70)
    print(f"LOUDNESS CHECK  target {a.target} LUFS")
    print("=" * 70)
    print(f"  {'condition':14s} {'n':>3s} {'LUFS before':>22s} {'LUFS after':>18s}")
    for cond, g in df.groupby("condition", sort=False):
        print(f"  {cond:14s} {len(g):3d} "
              f"{g.lufs_before.min():8.2f} to {g.lufs_before.max():8.2f} "
              f"{g.lufs_after.min():8.2f} to {g.lufs_after.max():8.2f}")
    spread = ok.lufs_after.max() - ok.lufs_after.min()
    print(f"\n  spread across all unclamped stimuli: {spread:.3f} dB")
    if spread > 0.5:
        print("  WARNING: level is still varying by more than 0.5 dB.")
    else:
        print("  Level is no longer confounded with condition.")

    manifest = dict(
        target_lufs=a.target, sample_rate=SR, max_sec=a.max_sec, seed=a.seed,
        n_speakers=len(chosen), speakers=[s.split("_")[0] for s in chosen],
        conditions=[dict(name=c, arm=arm or None, note=n) for c, arm, n in CONDITIONS],
        n_stimuli=len(rows), blinded=True,
        key_file="stimuli_blind_key.txt", key_csv=os.path.basename(key_csv),
        lufs_spread_db=round(float(spread), 3),
        clamped=int(df.clamped.sum()), warnings=warnings,
    )
    with open(os.path.join(a.out, "stimuli_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[stim] {len(rows)} stimuli -> {stim_dir}")
    print(f"[stim] key -> {key_csv}  (do NOT ship this with the audio)")
    if warnings:
        print(f"[stim] {len(warnings)} warnings:")
        for w in warnings[:10]:
            print("   ", w)


if __name__ == "__main__":
    main()
