"""
experiments/diag_directional_survival.py
========================================
Why does the d-vector lose protection under purification when WavLM-SV and ECAPA
do not?

THE OBSERVATION (RESULTS section 23)
------------------------------------
After fix P0, clone success with the loud budget, deployment setting:

    no purifier        WavLM 0 %   ECAPA 0 %   d-vector 12 %
    foreign purifier   WavLM 7 %   ECAPA 0 %   d-vector 68 %

So the perturbation fools the d-vector fine until a purifier runs, then stops --
while continuing to work for the other two. Purification is doing something
encoder-SPECIFIC, which a global "does the perturbation survive?" measurement
cannot see.

THE HYPOTHESIS, AND IT IS THE PAPER'S OWN
-----------------------------------------
`craft()`'s null_mode="full" asks the WHOLE perturbation to pass through. Its own
code comment says why that may be the wrong objective:

    "purification keeps delta large while absorbing its speaker-discriminative
     part"

If true, then delta survives (high signal gain) while the COMPONENT THAT MOVES A
GIVEN ENCODER does not -- and it could be absorbed more for one encoder than
another. That is measurable, and nothing in the project measures it yet: signal
gain is computed on delta as a whole, blind to which encoder cares about what.

WHAT THIS SCRIPT MEASURES
-------------------------
For each encoder e, the direction in mel space that actually moves it:

    g_e = d/d(mel) cos( E_e(V(mel)), E_e(x) )

Then it projects the crafted perturbation onto g_e and asks how much of THAT
projection survives purification:

    keep_e = <purified_shift, g_e> / <delta_mel, g_e>

keep_e near 1 -> the part of the perturbation that matters to encoder e comes
through. Near 0 -> the purifier absorbed exactly the part that mattered, which is
the mechanism the comment predicts and would explain the d-vector column.

Run:  python experiments/diag_directional_survival.py --n-files 4
"""
import sys, io, os, argparse

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=4, dest="n_files")
    ap.add_argument("--variant", default="fullnsT_m30_seg_all")
    ap.add_argument("--purifier-ckpt", default="checkpoints/purifier_v7.pt",
                    dest="ckpt")
    ap.add_argument("--t-diffuse", type=int, default=50, dest="t_diffuse")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.real_diffpure import load_real_diffpure
    from evaluation.evaluate import wav_to_logmel
    from experiments.exp_purifier_aware import (build_encoders, WAV_DIR,
                                                _build_mel_to_wav,
                                                _mel_to_wav_differentiable)
    from evaluation.evaluate import N_FFT, N_MELS, HOP_LENGTH, SAMPLE_RATE
    from experiments.librispeech import list_librispeech, reference_items, load_flac

    dp = load_real_diffpure(checkpoint_path=a.ckpt, t_diffuse=a.t_diffuse,
                            device=device, sampler="ddim")
    for p in dp.parameters():
        p.requires_grad_(False)
    encs = build_encoders(device)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)

    print(f"variant {a.variant}, purifier {os.path.basename(a.ckpt)}, t={a.t_diffuse}\n")
    print("keep_e = how much of the perturbation component that MOVES encoder e")
    print("survives purification. 1.0 = fully preserved, 0.0 = fully absorbed.\n")
    print(f"  {'speaker':<9}{'WavLM-SV':>11}{'ECAPA':>10}{'d-vector':>11}"
          f"{'global a':>11}")

    rows = []
    for spk in sorted(ref)[: a.n_files]:
        it = ref[spk]
        name = f"spk{spk}_{it['chapter']}_{it['utt']:04d}"
        p = os.path.join(WAV_DIR, f"{name}_{a.variant}.wav")
        if not os.path.isfile(p):
            continue
        wav_o = load_flac(it["path"], 4.0).to(device)
        y, _ = sf.read(p, dtype="float32")
        wav_p = torch.from_numpy(y).reshape(1, -1).to(device)[:, : wav_o.shape[-1]]

        mel_o = wav_to_logmel(wav_o, device).detach()
        mel_p = wav_to_logmel(wav_p, device).detach()
        d_mel = mel_p - mel_o

        with torch.no_grad():
            mel_pur = dp(mel_p, seed=1)
            shift = mel_pur - dp(mel_o, seed=1)

        inv, phase, win, wav_len, T_frames = _build_mel_to_wav(
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            sample_rate=SAMPLE_RATE, x_wav=wav_o, device=device)

        keeps = []
        for e in encs:
            with torch.no_grad():
                o = e.embed(wav_o).detach()
            m = mel_o.clone().requires_grad_(True)
            w = _mel_to_wav_differentiable(m, inv, phase, win, N_FFT,
                                           HOP_LENGTH, T_frames, wav_len)
            w = w / (w.abs().max() + 1e-6)
            c = (e.embed(w) * o).sum()
            g, = torch.autograd.grad(c, m)
            g = g / (g.norm() + 1e-9)
            num = float((shift * g).sum())
            den = float((d_mel * g).sum())
            keeps.append(num / den if abs(den) > 1e-9 else float("nan"))

        alpha = float((shift * d_mel).sum() / (d_mel * d_mel).sum())
        rows.append(keeps + [alpha])
        print(f"  {spk:<9}{keeps[0]:>11.3f}{keeps[1]:>10.3f}{keeps[2]:>11.3f}"
              f"{alpha:>11.3f}")

    if rows:
        m = np.nanmean(np.array(rows), axis=0)
        print(f"\n  {'MEAN':<9}{m[0]:>11.3f}{m[1]:>10.3f}{m[2]:>11.3f}{m[3]:>11.3f}")
        print("\nHOW TO READ IT")
        print("  d-vector keep MUCH lower than the others -> the purifier absorbs")
        print("    exactly the component that encoder cares about. That is the")
        print("    mechanism null_mode='directed' exists to fix, and it has never")
        print("    been run (every table on disk used null_mode='full').")
        print("  all three similar -> the cause is elsewhere; do NOT reach for")
        print("    'directed', and look at the encoder's own robustness instead.")
        print("  note 'global a' is signal gain on the WHOLE perturbation -- if it")
        print("    is high while a keep_e is low, that is the paper's own point:")
        print("    surviving as a signal is not the same as surviving as a threat.")


if __name__ == "__main__":
    main()
