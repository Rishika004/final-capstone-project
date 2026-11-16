"""experiments/make_step0_stimuli.py -- PLAN_V4 Step 0: the listening-frontier stimuli.

Builds the blinded, loudness-matched stimulus set that `PREREG_step0_listening_
frontier.md` (as amended 2026-10-10, §A1) specifies, from the SAME transform
code Step 1 measures (`utils/onmanifold.py`), so the frontier a listener hears
and the protection the probe measures are produced by one implementation.

THE DESIGN IN ONE PARAGRAPH
---------------------------
Six speakers from the 29-speaker cross-session cohort (3 F, 3 M), each with the
shipping additive arm on disk. Per speaker, every trial pairs a clean REFERENCE
clip from one recording session (the speaker's second chapter) with a TEST clip
from a different session (the utterance the additive arm protected, in the
largest chapter), so "same speaker?" is the cross-session question the
verifier answers, not clip matching. The test clip is the same utterance in
every condition; only the processing differs:

    original        the clean test utterance                (high anchor)
    anchor_diff     a different, same-sex speaker's clean utterance (low anchor)
    tilt_6db        -6 dB spectral tilt                     (0 % protection)
    mcadams_100     LPC analysis-resynthesis, alpha = 1.00  (processing control)
    mcadams_095 / _090 / _085 / _080                        (the frontier rungs)
    additive_eot    fullnsT_tc_m30_seg, the shipping arm    (the incumbent)

9 conditions x 6 speakers = 54 main trials, plus 6 `original` repeats (a second
utterance from the test session) and 6 `anchor_diff` repeats (a second
impostor) for the exclusion check = 66 trials per listener.

BLINDING AND LEVEL
------------------
Every file is `stim_<16 hex>.wav`, hex = HMAC(key, speaker|role|condition|utt);
the key is written separately and is not read until analysis. Every clip,
reference and test alike, is normalised to -23 LUFS (BS.1770-4,
`utils/loudness.py`); any clip the peak ceiling stopped short is reported.

Outputs, under `listening_test/step0/`:
    stimuli/stim_<hash>.wav      blinded audio (references and tests)
    stimuli_key.csv              hash -> speaker, role, condition, source, LUFS
    trials.json                  the 66 trials as (ref_stim, test_stim) pairs
    stimuli_blind_key.txt        the HMAC key (do not ship)
    manifest.json                build record, transform stats, warnings

Run:
    python experiments/make_step0_stimuli.py
    python experiments/make_step0_stimuli.py --n-speakers 2 --out /tmp/x   # smoke
"""
import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.librispeech import (cross_session_split, list_librispeech,  # noqa: E402
                                     load_flac, reference_items)
from utils.loudness import normalise  # noqa: E402
from utils.onmanifold import ARMS, apply_arm  # noqa: E402

WAV_DIR = os.path.join(PROJ, "experiments", "results", "purifier_aware")
LIBRI = os.path.join(PROJ, "data", "librispeech")
OUT_DIR = os.path.join(PROJ, "listening_test", "step0")
SR = 16000
ADDITIVE_TAG = "fullnsT_tc_m30_seg"

# Order is the order the analysis prints. The condition name never reaches the
# page or the listener.
CONDITIONS = ["original", "anchor_diff", "tilt_6db", "mcadams_100", "mcadams_095",
              "mcadams_090", "mcadams_085", "mcadams_080", "additive_eot"]
MCADAMS = [c for c in CONDITIONS if c.startswith("mcadams")]


def speaker_sex(root):
    """{speaker_id: 'F'|'M'} from LibriSpeech's SPEAKERS.TXT."""
    out = {}
    with open(os.path.join(root, "LibriSpeech", "SPEAKERS.TXT"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(";") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            try:
                out[int(parts[0])] = parts[1]
            except ValueError:
                continue
    return out


def blind_name(key, *fields):
    return hmac.new(key, "|".join(map(str, fields)).encode(), hashlib.sha256).hexdigest()[:16]


def crafted_stems(tag):
    """speaker -> 'spk1089_134686_0000' for every crafted file of `tag`."""
    out = {}
    suf = f"_{tag}.wav"
    for f in os.listdir(WAV_DIR):
        if f.endswith(suf):
            stem = f[: -len(suf)]
            out[int(stem.split("_")[0].replace("spk", ""))] = stem
    return out


def pick_speakers(cands, sex, n, rng):
    """n speakers, as balanced by sex as n allows, deterministic in rng."""
    f = sorted(s for s in cands if sex.get(s) == "F")
    m = sorted(s for s in cands if sex.get(s) == "M")
    nf = n // 2
    nm = n - nf
    if len(f) < nf or len(m) < nm:
        raise SystemExit(f"not enough speakers per sex: F={len(f)} M={len(m)}")
    chosen = sorted(rng.choice(f, nf, replace=False).tolist()
                    + rng.choice(m, nm, replace=False).tolist())
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-speakers", type=int, default=6, dest="n_speakers")
    ap.add_argument("--target-lufs", type=float, default=-23.0, dest="target")
    ap.add_argument("--max-sec", type=float, default=4.0, dest="max_sec")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    items = list_librispeech(LIBRI, "test-clean", min_sec=a.max_sec)
    split = cross_session_split(items, min_sec=a.max_sec)
    ref = reference_items(items, min_sec=a.max_sec, utt_index=0)
    sex = speaker_sex(LIBRI)
    stems = crafted_stems(ADDITIVE_TAG)

    # Eligible = cross-session cohort with the shipping arm on disk, and whose
    # crafted utterance really is in the reference (largest) chapter.
    cohort = []
    for s, stem in sorted(stems.items()):
        if s not in split or s not in ref:
            continue
        exp = f"spk{s}_{ref[s]['chapter']}_{int(ref[s]['utt']):04d}"
        if exp != stem:
            print(f"[step0] WARNING {stem} is not the shared selector's reference "
                  f"({exp}); skipping speaker {s}")
            continue
        cohort.append(s)
    print(f"[step0] {len(cohort)} eligible cohort speakers with {ADDITIVE_TAG}")

    chosen = pick_speakers(cohort, sex, a.n_speakers, rng)
    others = [s for s in cohort if s not in chosen]
    print(f"[step0] chosen: {', '.join(f'{s}{sex[s]}' for s in chosen)}")

    # Two impostors per chosen speaker, same sex, from the rest of the cohort,
    # each used as rarely as possible so no impostor voice recurs more than it must.
    impostor_use = {s: 0 for s in others}

    def impostors(s, k=2):
        pool = sorted((impostor_use[o], rng.random(), o) for o in others
                      if sex[o] == sex[s])
        picked = [o for _, _, o in pool[:k]]
        for o in picked:
            impostor_use[o] += 1
        return picked

    key = secrets.token_bytes(32)
    stim_dir = os.path.join(a.out, "stimuli")
    os.makedirs(stim_dir, exist_ok=True)
    win = torch.hann_window(1024)

    rows, trials, warnings, tstats = [], [], [], []
    written = {}

    def emit(spk, role, cond, utt_id, x, src, extra=None):
        """Normalise, blind, write. Returns the stim id (reused if identical)."""
        h = blind_name(key, spk, role, cond, utt_id)
        if h in written:
            return h
        y, info = normalise(np.asarray(x, dtype=np.float64), SR, target_lufs=a.target)
        if info["clamped"]:
            warnings.append(f"{spk}/{role}/{cond}/{utt_id}: peak clamped at "
                            f"{info['lufs_after']} LUFS, not {a.target}")
        sf.write(os.path.join(stim_dir, f"stim_{h}.wav"), y, SR)
        rows.append(dict(stim_id=h, speaker=spk, role=role, condition=cond,
                         utterance=utt_id, source=os.path.relpath(src, PROJ),
                         lufs_before=info["lufs_before"], gain_db=info["gain_db"],
                         lufs_after=info["lufs_after"], peak=info["peak"],
                         clamped=info["clamped"], **(extra or {})))
        written[h] = True
        return h

    def utt_id(it):
        return f"{it['speaker']}-{it['chapter']}-{it['utt']:04d}"

    for s in chosen:
        test_it = ref[s]                               # session A, the crafted utterance
        ref_it = split[s]["enroll"][0]                 # session B, clean reference
        alt = [it for it in split[s]["ref"] if utt_id(it) != utt_id(test_it)]
        if not alt:
            raise SystemExit(f"speaker {s} has no second utterance in its test session")
        alt_it = alt[0]

        wav_o = load_flac(test_it["path"], a.max_sec)
        ref_h = emit(s, "reference", "clean_ref", utt_id(ref_it),
                     load_flac(ref_it["path"], a.max_sec).squeeze(0).numpy(), ref_it["path"])

        for cond in CONDITIONS:
            if cond == "original":
                x, src, extra = wav_o.squeeze(0).numpy(), test_it["path"], {}
                tid = utt_id(test_it)
                spk_of_test = s
            elif cond == "anchor_diff":
                imp = impostors(s)
                imp_it = ref[imp[0]]
                x = load_flac(imp_it["path"], a.max_sec).squeeze(0).numpy()
                src, extra, tid, spk_of_test = imp_it["path"], dict(impostor=imp[0]), utt_id(imp_it), imp[0]
                # the repeat, second impostor
                imp2_it = ref[imp[1]]
                h2 = emit(imp[1], "test", "anchor_diff", utt_id(imp2_it),
                          load_flac(imp2_it["path"], a.max_sec).squeeze(0).numpy(),
                          imp2_it["path"], dict(impostor=imp[1], paired_with=s))
                trials.append(dict(block="anchor_repeat", speaker=s, ref_stim=ref_h, test_stim=h2))
            elif cond == "additive_eot":
                p = os.path.join(WAV_DIR, f"{stems[s]}_{ADDITIVE_TAG}.wav")
                d, sr2 = sf.read(p, dtype="float32")
                if sr2 != SR:
                    warnings.append(f"{p}: {sr2} Hz")
                x = d[: wav_o.shape[-1]]
                src, extra, tid, spk_of_test = p, dict(arm=ADDITIVE_TAG), utt_id(test_it), s
            else:
                st = {}
                y = apply_arm(cond, wav_o, win, stats=st).clamp(-1, 1)
                x = y.squeeze(0).numpy()[: wav_o.shape[-1]]
                src, extra, tid, spk_of_test = test_it["path"], dict(**st), utt_id(test_it), s
                tstats.append(dict(speaker=s, condition=cond, **st))
            h = emit(spk_of_test, "test", cond, tid, x, src, extra)
            trials.append(dict(block="main", speaker=s, ref_stim=ref_h, test_stim=h))

        # `original` repeat: a second clean utterance from the test session.
        h_alt = emit(s, "test", "original", utt_id(alt_it),
                     load_flac(alt_it["path"], a.max_sec).squeeze(0).numpy(), alt_it["path"])
        trials.append(dict(block="anchor_repeat", speaker=s, ref_stim=ref_h, test_stim=h_alt))
        print(f"  spk{s} ({sex[s]}): {len(CONDITIONS)} conditions, ref={utt_id(ref_it)}, "
              f"test={utt_id(test_it)}", flush=True)

    for i, t in enumerate(trials):
        t["trial_id"] = f"t{i:02d}"

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "stimuli_key.csv"), index=False)
    with open(os.path.join(a.out, "stimuli_blind_key.txt"), "w") as fh:
        fh.write(key.hex() + "\n")
    with open(os.path.join(a.out, "trials.json"), "w", encoding="utf-8") as fh:
        json.dump([dict(trial_id=t["trial_id"], block=t["block"], ref=f"stimuli/stim_{t['ref_stim']}.wav",
                        test=f"stimuli/stim_{t['test_stim']}.wav") for t in trials],
                  fh, indent=0)

    # Verification printed, because a silent normaliser is worth nothing.
    ok = df[~df.clamped]
    spread = float(ok.lufs_after.max() - ok.lufs_after.min()) if len(ok) else float("nan")
    print("\n" + "=" * 70)
    print(f"LOUDNESS CHECK  target {a.target} LUFS   spread (unclamped) {spread:.3f} dB")
    print("=" * 70)
    for cond, g in df.groupby("condition", sort=False):
        print(f"  {cond:12s} n={len(g):2d}  before {g.lufs_before.min():7.2f}..{g.lufs_before.max():7.2f}"
              f"  after {g.lufs_after.min():7.2f}..{g.lufs_after.max():7.2f}")
    if spread > 0.5:
        print("  WARNING: level still varies by more than 0.5 dB")

    ts = pd.DataFrame(tstats)
    if len(ts):
        print("\nTRANSFORM FRAME STATS (silent frames are passed through unchanged)")
        print(ts.groupby("condition")[["n_frames", "n_silent", "n_failed", "n_unstable"]]
                .sum().to_string())

    n_main = sum(t["block"] == "main" for t in trials)
    n_rep = len(trials) - n_main
    manifest = dict(
        prereg="paper/PREREG_step0_listening_frontier.md", amendment="A1 2026-10-10",
        target_lufs=a.target, sample_rate=SR, max_sec=a.max_sec, seed=a.seed,
        speakers=[dict(id=s, sex=sex[s]) for s in chosen], conditions=CONDITIONS,
        additive_tag=ADDITIVE_TAG, n_stimuli=len(rows), n_trials=len(trials),
        n_main=n_main, n_anchor_repeats=n_rep, blinded=True,
        key_file="stimuli_blind_key.txt", key_csv="stimuli_key.csv",
        lufs_spread_db=round(spread, 3), clamped=int(df.clamped.sum()),
        transform_stats=ts.to_dict("records") if len(ts) else [], warnings=warnings)
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[step0] {len(rows)} stimuli, {len(trials)} trials ({n_main} main + {n_rep} anchor repeats)")
    print(f"[step0] -> {a.out}   key: stimuli_key.csv (do NOT ship with the audio)")
    if warnings:
        print(f"[step0] {len(warnings)} warning(s):")
        for w in warnings[:10]:
            print("   ", w)


if __name__ == "__main__":
    main()
