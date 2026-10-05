"""
experiments/diag_dvec_proxy.py
==============================
P0 DIAGNOSTIC. Where does the Resemblyzer proxy gap come from?

THE PROBLEM
-----------
In the DEPLOYMENT setting (all three encoders crafted against), Resemblyzer still
lets 35-50 % of clones through, while WavLM-SV and ECAPA reach 0 %. It is being
optimised against, so this is NOT encoder transfer failing. We are optimising
against a different function than the one that scores us.

`DvecEnc.embed()`        torch mel -> dvec(mel) -> one pooled embedding
`real_embed_dvec()`      preprocess_wav() -> embed_utterance()
                         = volume normalise + trim silences, then SPLIT into
                           1.6 s partials, embed each, L2-normed AVERAGE

Two candidate causes, and the fix differs depending on which dominates:

  (a) PARTIAL AVERAGING. AntiFake Sec. C2 predicts this one: "directly deviating
      the final embedding is susceptible to local optima, as the impact of a
      partial embedding not close to the target is mitigated by other partials".
      Fix = score every partial, push the worst.

  (b) PREPROCESSING. Volume normalisation + voice-activity trimming change the
      signal before it is ever embedded. Fix = model it, or craft on the
      preprocessed signal.

This script measures both, separately, before any code is written.

Run:  python experiments/diag_dvec_proxy.py
"""
import sys, os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import torch
import torch.nn.functional as F


def l2(x):
    return F.normalize(x.float().reshape(1, -1), dim=-1)


def cos(a, b):
    return float((l2(a) * l2(b)).sum())


def main():
    device = torch.device("cpu")          # GPU is busy; this is tiny
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.hparams import partials_n_frames, mel_window_step, sampling_rate
    from experiments.librispeech import list_librispeech, reference_items, load_flac
    from experiments.multi_encoders import DvecEnc

    dvec = VoiceEncoder(device=device)
    proxy = DvecEnc(dvec, device)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)
    spks = sorted(ref)[:6]

    print("Cosine between the embedding the OPTIMISER sees and the one that SCORES us.")
    print("1.0 would mean no gap at all. Lower = we are optimising the wrong thing.\n")
    print(f"{'speaker':<10}{'proxy vs real':>15}{'proxy vs partials':>20}"
          f"{'partials vs real':>19}{'n_partials':>12}")

    rows = []
    for spk in spks:
        wav_np = load_flac(ref[spk]["path"], 4.0).squeeze(0).numpy().astype(np.float32)
        wav_t = torch.from_numpy(wav_np).reshape(1, -1)

        # (1) what the optimiser currently sees
        with torch.no_grad():
            e_proxy = proxy.embed(wav_t)

        # (2) the real scoring path: preprocess + partial average
        pp = preprocess_wav(wav_np, source_sr=16000)
        e_real = torch.tensor(dvec.embed_utterance(pp)).reshape(1, -1)

        # (3) partial averaging WITHOUT preprocessing -- isolates cause (a)
        samples_per_frame = int(sampling_rate * mel_window_step / 1000)
        wav_slices, mel_slices = dvec.compute_partial_slices(
            len(wav_np), rate=1.3, min_coverage=0.75)   # embed_utterance defaults
        pad = wav_slices[-1].stop - len(wav_np)
        w = np.pad(wav_np, (0, max(0, pad)), "constant")
        wt = torch.from_numpy(w).reshape(1, -1)
        with torch.no_grad():
            mel = proxy.mel(wt.float()).squeeze(0).transpose(0, 1)      # [frames, 40]
            parts = torch.stack([mel[s] for s in mel_slices])            # [n, 160, 40]
            pe = dvec(parts)                                             # [n, 256]
            e_parts = l2(F.normalize(pe, dim=-1).mean(dim=0))

        rows.append((cos(e_proxy, e_real), cos(e_proxy, e_parts),
                     cos(e_parts, e_real), len(mel_slices)))
        print(f"{spk:<10}{rows[-1][0]:>15.3f}{rows[-1][1]:>20.3f}"
              f"{rows[-1][2]:>19.3f}{rows[-1][3]:>12d}")

    a = np.array([r[:3] for r in rows]).mean(axis=0)
    print(f"\n{'MEAN':<10}{a[0]:>15.3f}{a[1]:>20.3f}{a[2]:>19.3f}")
    print("\nHOW TO READ THIS")
    print("  'proxy vs partials' LOW  -> cause (a), partial averaging. Fix: score")
    print("                              every partial, push the worst (AntiFake C2).")
    print("  'partials vs real'  LOW  -> cause (b), preprocessing (normalise + VAD).")
    print("  both high but 'proxy vs real' low -> the two causes compound.")


if __name__ == "__main__":
    main()
