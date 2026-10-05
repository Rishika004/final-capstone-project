"""
experiments/exp_robustness.py
=============================
Real-world robustness of null-space protection.

We take the protected audio and subject it to common real-world
transformations an attacker (or a transmission channel) might apply
BEFORE running the cloning pipeline, then re-measure whether the
protection survives:

  - Gaussian noise addition        (sigma = 0.005, 0.01)
  - Down/up resampling             (16k -> 8k -> 16k)
  - Bit-depth quantization         (8-bit)
  - Low-pass filtering             (cutoff 4 kHz)

For each transform we report:
  - WavLM cosine similarity  cos(orig, transformed_protected)   (< 0.5 = cloning still fails)
  - PRR                        (protection survival in mel space)

True MP3 transcoding needs ffmpeg (missing on this machine); the
quantization + low-pass + resample chain is a faithful lossy-codec proxy.

Run:
  python experiments/exp_robustness.py --n-files 3
Output:
  experiments/results/robustness.csv
"""

import sys, io, os, argparse, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Make project root importable
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
import numpy as np
import pandas as pd
import torchaudio.functional as AF

from core.perturbation import apply_null_space_protection
from evaluation.evaluate import load_wav, wav_to_logmel, SAMPLE_RATE
from evaluation.metrics import speaker_cosine_similarity, purification_resistance_rate
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_speaker_encoder

OUT_DIR = os.path.join(PROJ, "experiments", "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Real-world transforms (all operate on a [1, T] waveform tensor, 16 kHz)
# ---------------------------------------------------------------------------

def t_identity(wav, device):
    return wav.clone()

def t_gaussian(sigma):
    def fn(wav, device):
        torch.manual_seed(0)
        return (wav + torch.randn_like(wav) * sigma).clamp(-1.0, 1.0)
    return fn

def t_resample_8k(wav, device):
    """16k -> 8k -> 16k round-trip (bandwidth loss like a phone channel)."""
    down = AF.resample(wav, SAMPLE_RATE, 8000)
    up   = AF.resample(down, 8000, SAMPLE_RATE)
    if up.shape[-1] > wav.shape[-1]:
        up = up[..., :wav.shape[-1]]
    elif up.shape[-1] < wav.shape[-1]:
        up = torch.nn.functional.pad(up, (0, wav.shape[-1] - up.shape[-1]))
    return up.clamp(-1.0, 1.0)

def t_quantize_8bit(wav, device):
    """8-bit uniform quantization (lossy codec proxy)."""
    levels = 2 ** 8
    q = torch.round((wav + 1.0) * 0.5 * (levels - 1))
    return (q / (levels - 1) * 2.0 - 1.0).clamp(-1.0, 1.0)

def t_lowpass_4k(wav, device):
    """Low-pass filter at 4 kHz (removes high-frequency perturbation energy)."""
    return AF.lowpass_biquad(wav, SAMPLE_RATE, cutoff_freq=4000.0).clamp(-1.0, 1.0)


TRANSFORMS = {
    "none (baseline)":       t_identity,
    "gaussian sigma=0.005":  t_gaussian(0.005),
    "gaussian sigma=0.01":   t_gaussian(0.01),
    "resample 16k-8k-16k":   t_resample_8k,
    "quantize 8-bit":        t_quantize_8bit,
    "lowpass 4kHz":          t_lowpass_4k,
}


def protect_one(wav_path, purifier, speaker_encoder, device, epsilon, K, pgd_steps):
    """Return (wav_original, wav_protected, mel_original, mel_protected)."""
    wav_original = load_wav(wav_path).to(device)
    mel_original = wav_to_logmel(wav_original, device)

    mel_protected, best_wav_delta = apply_null_space_protection(
        x_mel=mel_original, purifier=purifier, epsilon=epsilon, K=K,
        mode="pgd", x_wav=wav_original, speaker_encoder=speaker_encoder,
        pgd_steps=pgd_steps, verify=False,
    )

    orig_len = wav_original.shape[-1]
    with torch.no_grad():
        d = best_wav_delta.to(device)
        if d.shape[-1] > orig_len:
            d = d[..., :orig_len]
        elif d.shape[-1] < orig_len:
            d = torch.nn.functional.pad(d, (0, orig_len - d.shape[-1]))
        wav_protected = (wav_original + d).clamp(-1.0, 1.0)

    return wav_original, wav_protected, mel_original, mel_protected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--n-files", type=int, default=3, dest="n_files")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--pgd-steps", type=int, default=50, dest="pgd_steps")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Robustness] device={device}")

    from pathlib import Path
    wavs = sorted(Path(args.input).glob("*.wav"))[:args.n_files]
    print(f"[Robustness] {len(wavs)} file(s)")

    print("[Robustness] loading models ...")
    purifier = load_real_diffpure(t_diffuse=50, device=device)
    speaker_encoder = load_speaker_encoder(device=device)

    rows = []
    for wp in wavs:
        print(f"\n{'='*60}\n[Robustness] protecting {wp.name}")
        wav_o, wav_p, mel_o, mel_p = protect_one(
            str(wp), purifier, speaker_encoder, device,
            args.epsilon, args.K, args.pgd_steps,
        )
        # Baseline: cos-sim of clean protected vs original (no transform)
        for tname, tfn in TRANSFORMS.items():
            wav_pt = tfn(wav_p, device)
            cos = speaker_cosine_similarity(wav_o, wav_pt, speaker_encoder)
            # PRR: re-mel the transformed protected audio, compare purifier residuals
            mel_pt = wav_to_logmel(wav_pt, device)
            prr = purification_resistance_rate(mel_o, mel_pt, purifier)
            survives = "YES" if cos < 0.5 else "NO"
            rows.append({
                "file": wp.name, "transform": tname,
                "cos_sim_after": round(cos, 4),
                "prr": round(prr, 4),
                "cloning_still_fails": survives,
            })
            print(f"  {tname:24s} | cos_sim={cos:+.4f} | PRR={prr:.4f} | fails={survives}")

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "robustness.csv")
    df.to_csv(csv, index=False)
    print(f"\n[Robustness] saved {csv}")

    # Aggregate: mean cos-sim per transform across files
    print("\n" + "=" * 60)
    print("MEAN COSINE SIMILARITY PER TRANSFORM (across files)")
    print("=" * 60)
    agg = df.groupby("transform", sort=False).agg(
        mean_cos=("cos_sim_after", "mean"),
        mean_prr=("prr", "mean"),
        pct_fail=("cloning_still_fails", lambda s: 100.0 * (s == "YES").mean()),
    )
    for tname, r in agg.iterrows():
        print(f"  {tname:24s} | mean cos={r.mean_cos:+.4f} | "
              f"mean PRR={r.mean_prr:.4f} | protection holds on {r.pct_fail:.0f}% of files")
    print("=" * 60)


if __name__ == "__main__":
    main()
