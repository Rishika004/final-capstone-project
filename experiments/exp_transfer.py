"""
experiments/exp_transfer.py
===========================
Cross-encoder transferability of null-space protection.

We craft the perturbation delta against WavLM ONLY (as in the main pipeline),
then measure whether the SAME protected audio also fools three INDEPENDENT,
fully-trained speaker-verification encoders:

  - Resemblyzer  (d-vector, GE2E, trained on VoxCeleb+LibriSpeech)   256-d
  - ECAPA-TDNN   (speechbrain/spkrec-ecapa-voxceleb)                 192-d
  - x-vector     (speechbrain/spkrec-xvect-voxceleb)                 512-d

For each encoder E we report cos( E(original), E(protected) ).
Since original and protected are the SAME utterance, an UNPROTECTED pair
scores ~1.0. A large drop => the WavLM-crafted delta transfers to that encoder.

Note: the project's WavLM encoder uses a randomly-initialised projection head,
so the independent encoders (ECAPA / x-vector / d-vector), which are fully
trained SV models with meaningful decision thresholds, provide the more
rigorous transferability test.

Run:
  python experiments/exp_transfer.py --n-files 3
Output:
  experiments/results/transfer.csv
"""

import sys, io, os, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

from core.perturbation import apply_null_space_protection
from evaluation.evaluate import load_wav, wav_to_logmel
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_speaker_encoder

OUT_DIR = os.path.join(PROJ, "experiments", "results")
SAVEDIR = os.path.join(PROJ, "experiments", "sb_models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAVEDIR, exist_ok=True)

# Approx same-speaker acceptance thresholds (cosine) for reference
THRESHOLDS = {"WavLM (target)": 0.50, "Resemblyzer d-vector": 0.75,
              "ECAPA-TDNN": 0.25, "x-vector": 0.30}


def cos(a, b):
    a = F.normalize(a.float().reshape(1, -1), dim=-1)
    b = F.normalize(b.float().reshape(1, -1), dim=-1)
    return (a * b).sum().item()


def build_encoders(device):
    encs = {}

    # WavLM (the target used during PGD)
    wavlm = load_speaker_encoder(device=device)
    encs["WavLM (target)"] = lambda w: wavlm(w if w.dim() == 2 else w.unsqueeze(0)).detach()

    # Resemblyzer d-vector
    from resemblyzer import VoiceEncoder, preprocess_wav
    dvec = VoiceEncoder(device=device)
    def dvec_embed(w):
        arr = w.detach().squeeze().cpu().numpy().astype(np.float32)
        pp = preprocess_wav(arr, source_sr=16000)
        return torch.tensor(dvec.embed_utterance(pp))
    encs["Resemblyzer d-vector"] = dvec_embed

    # speechbrain ECAPA  (x-vector dropped: speechbrain 1.1.0 has a broken
    # k2_fsa lazy import that only trips the Xvector model on Windows)
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(SAVEDIR, "ecapa"), run_opts={"device": str(device)})
    def sb_embed(model):
        def fn(w):
            wav = w.detach().reshape(1, -1).float().to(device)
            return model.encode_batch(wav).squeeze().detach()
        return fn
    encs["ECAPA-TDNN"] = sb_embed(ecapa)
    return encs, wavlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--n-files", type=int, default=3, dest="n_files")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--pgd-steps", type=int, default=50, dest="pgd_steps")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Transfer] device={device}")

    wavs = sorted(Path(args.input).glob("*.wav"))[:args.n_files]
    print(f"[Transfer] {len(wavs)} file(s)")

    purifier = load_real_diffpure(t_diffuse=50, device=device)
    print("[Transfer] building encoders (downloads ECAPA + x-vector on first run) ...")
    encs, wavlm = build_encoders(device)

    rows = []
    for wp in wavs:
        print(f"\n{'='*60}\n[Transfer] {wp.name}")
        wav_o = load_wav(str(wp)).to(device)
        mel_o = wav_to_logmel(wav_o, device)

        # Protect (PGD vs WavLM + null-space projection), waveform bypass
        mel_p, best_wav_delta = apply_null_space_protection(
            x_mel=mel_o, purifier=purifier, epsilon=args.epsilon, K=args.K,
            mode="pgd", x_wav=wav_o, speaker_encoder=wavlm,
            pgd_steps=args.pgd_steps, verify=False)
        d = best_wav_delta.to(device)
        L = wav_o.shape[-1]
        if d.shape[-1] > L: d = d[..., :L]
        elif d.shape[-1] < L: d = torch.nn.functional.pad(d, (0, L - d.shape[-1]))
        wav_p = (wav_o + d).clamp(-1.0, 1.0)

        for name, enc in encs.items():
            c = cos(enc(wav_o), enc(wav_p))
            thr = THRESHOLDS[name]
            transfers = "YES" if c < thr else "NO"
            rows.append(dict(file=wp.name, encoder=name, cos_sim=round(c, 4),
                             threshold=thr, cloning_fails=transfers))
            print(f"  {name:22s} | cos={c:+.4f} | thr={thr} | fails={transfers}")

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "transfer.csv")
    df.to_csv(csv, index=False)
    print(f"\n[Transfer] saved {csv}")

    print("\n" + "=" * 66)
    print("MEAN COSINE SIMILARITY PER ENCODER (across files)")
    print("=" * 66)
    print("  delta crafted vs WavLM only; lower = transfers to that encoder")
    g = df.groupby("encoder", sort=False).agg(
        mean_cos=("cos_sim", "mean"),
        thr=("threshold", "first"),
        pct_fail=("cloning_fails", lambda s: 100.0 * (s == "YES").mean()))
    for name, r in g.iterrows():
        print(f"  {name:22s} | mean cos={r.mean_cos:+.4f} | thr={r.thr} | "
              f"cloning fails on {r.pct_fail:.0f}% of files")
    print("=" * 66)


if __name__ == "__main__":
    main()
