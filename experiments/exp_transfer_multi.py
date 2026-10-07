"""
experiments/exp_transfer_multi.py
=================================
Ensemble PGD: craft ONE waveform perturbation that fools WavLM + ECAPA +
Resemblyzer d-vector jointly, so cloning fails on all three (transferable).

Loss (minimised, signed-gradient PGD in waveform space):
    L = mean_i  cos( E_i(wav+delta), E_i(wav) )           [--agg mean]
  or  L = max_i  cos( E_i(wav+delta), E_i(wav) )           [--agg max]

The max aggregation targets the hardest (least-fooled) encoder each step,
which is usually better for ensembles.

Evaluation uses the REAL (eval-time) encoder paths, not the differentiable
proxies, so the reported numbers are honest.

DiffPure survival (PRR) of the resulting audio is also reported via the
existing null-space pipeline (mel delta projected onto null(J)).

Run:
  python experiments/exp_transfer_multi.py --n-files 3 --epsilon-wav 0.05 --steps 80 --agg max
Output:
  experiments/results/transfer_multi.csv
"""
import sys, io, os, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path: sys.path.insert(0, PROJ)

import torch, torch.nn.functional as F
import numpy as np, pandas as pd
from pathlib import Path

from core.jacobian import estimate_jacobian_randomized
from core.nullspace import compute_null_space_factored, project_onto_null_space
from evaluation.evaluate import load_wav, wav_to_logmel
from evaluation.metrics import purification_resistance_rate, signal_to_noise_ratio
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_wavlm
from experiments.multi_encoders import (build_all, WavLMEnc, ECAPAEnc, DvecEnc,
                                        real_embed_wavlm, real_embed_ecapa, real_embed_dvec)

OUT_DIR = os.path.join(PROJ, "experiments", "results")
SAVEDIR = os.path.join(PROJ, "experiments", "sb_models")
os.makedirs(OUT_DIR, exist_ok=True)


def ensemble_pgd(wav_o, encoders, epsilon_wav, n_steps, agg, weights, seed=42):
    """Signed-gradient PGD against multiple encoders. Returns best_wav_delta."""
    device = wav_o.device
    orig = []
    for e in encoders:
        with torch.no_grad():
            orig.append(e.embed(wav_o).detach())

    torch.manual_seed(seed)
    delta = ((torch.rand_like(wav_o) * 2 - 1) * epsilon_wav * 0.1).detach().requires_grad_(True)
    step = epsilon_wav / max(n_steps // 2, 1)

    best_score = float("inf")
    best_delta = delta.detach().clone()
    print(f"  ensemble PGD | agg={agg} | eps_wav={epsilon_wav} | steps={n_steps} | "
          f"encoders={[e.name for e in encoders]}")
    for s in range(1, n_steps + 1):
        if delta.grad is not None:
            delta.grad.zero_()
        wav_p = wav_o.detach() + delta
        coss = []
        for e, o in zip(encoders, orig):
            coss.append((e.embed(wav_p) * o).sum())
        coss_t = torch.stack(coss)
        wv = torch.tensor(weights, device=device, dtype=coss_t.dtype)
        if agg == "max":
            # smooth-max via weighted logsumexp keeps all encoders in the gradient
            loss = torch.logsumexp(coss_t * 4.0, dim=0) / 4.0
        else:
            loss = (coss_t * wv).sum() / wv.sum()
        loss.backward()
        with torch.no_grad():
            delta.data -= step * delta.grad.sign()
            delta.data.clamp_(-epsilon_wav, epsilon_wav)
        # score = worst-case (max) cosine — we want to minimise the max
        score = coss_t.max().item()
        if score < best_score:
            best_score = score
            best_delta = delta.detach().clone()
        if s % 10 == 0 or s == 1:
            with torch.no_grad():
                cs = ", ".join(f"{e.name.split()[0]}={c.item():+.2f}"
                               for e, c in zip(encoders, coss))
            print(f"    step {s:3d}/{n_steps} | {cs} | max={score:+.3f}")
    print(f"  done | best max-cos={best_score:+.3f}")
    return best_delta.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--n-files", type=int, default=3, dest="n_files")
    ap.add_argument("--epsilon-wav", type=float, default=0.05, dest="epsilon_wav")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--agg", choices=["mean", "max"], default="max")
    ap.add_argument("--weights", default="1,1,1", help="comma weights wavlm,ecapa,dvec")
    ap.add_argument("--K", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = [float(x) for x in args.weights.split(",")]
    print(f"[TransferMulti] device={device} | agg={args.agg} | weights={weights}")

    wavs = sorted(Path(args.input).glob("*.wav"))[:args.n_files]
    purifier = load_real_diffpure(t_diffuse=50, device=device)
    wavlm = load_wavlm(device=device)
    encoders = build_all(device, wavlm, SAVEDIR)
    ecapa_clf = [e for e in encoders if isinstance(e, ECAPAEnc)][0].clf
    dvec_model = [e for e in encoders if isinstance(e, DvecEnc)][0].dvec

    rows = []
    for wp in wavs:
        print(f"\n{'='*60}\n[TransferMulti] {wp.name}")
        wav_o = load_wav(str(wp)).to(device)
        mel_o = wav_to_logmel(wav_o, device)

        # ensemble PGD
        best_delta = ensemble_pgd(wav_o, encoders, args.epsilon_wav,
                                  args.steps, args.agg, weights)
        wav_p = (wav_o + best_delta).clamp(-1.0, 1.0)

        # DiffPure survival (PRR) via null-space projection of the mel delta
        Z, V = estimate_jacobian_randomized(purifier, mel_o, K=args.K, seed=42)
        range_basis, _ = compute_null_space_factored(Z, V, threshold=1e-3)
        mel_p = wav_to_logmel(wav_p, device)
        dmel = (mel_p - mel_o).cpu()
        dmel_proj = project_onto_null_space(dmel, range_basis, kind="range").to(device)
        prr = purification_resistance_rate(mel_o, mel_o + dmel_proj, purifier)
        snr = signal_to_noise_ratio(wav_o, wav_p)

        # REAL-path evaluation on all three encoders
        def cs(a, b):
            a = F.normalize(a.float().reshape(1, -1), dim=-1)
            b = F.normalize(b.float().reshape(1, -1), dim=-1)
            return (a * b).sum().item()
        cos_wavlm = cs(real_embed_wavlm(wavlm, wav_o), real_embed_wavlm(wavlm, wav_p))
        cos_ecapa = cs(real_embed_ecapa(ecapa_clf, wav_o), real_embed_ecapa(ecapa_clf, wav_p))
        cos_dvec = cs(real_embed_dvec(dvec_model, wav_o), real_embed_dvec(dvec_model, wav_p))

        for name, c, thr in [("WavLM (target)", cos_wavlm, 0.50),
                             ("ECAPA-TDNN", cos_ecapa, 0.25),
                             ("Resemblyzer d-vector", cos_dvec, 0.75)]:
            rows.append(dict(file=wp.name, encoder=name, cos_sim=round(c, 4),
                             threshold=thr, cloning_fails=("YES" if c < thr else "NO"),
                             prr=round(prr, 4), snr_db=round(snr, 2)))
        print(f"  REAL eval | WavLM={cos_wavlm:+.3f}(<0.5) "
              f"ECAPA={cos_ecapa:+.3f}(<0.25) dvec={cos_dvec:+.3f}(<0.75) "
              f"| PRR={prr:.3f} SNR={snr:.1f}dB")

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "transfer_multi.csv")
    df.to_csv(csv, index=False)
    print(f"\n[TransferMulti] saved {csv}")

    print("\n" + "=" * 70)
    print("MEAN PER ENCODER (real eval) — ensemble PGD")
    print("=" * 70)
    g = df.groupby("encoder", sort=False).agg(
        mean_cos=("cos_sim", "mean"), thr=("threshold", "first"),
        pct_fail=("cloning_fails", lambda s: 100.0 * (s == "YES").mean()))
    for name, r in g.iterrows():
        print(f"  {name:22s} | mean cos={r.mean_cos:+.4f} | thr={r.thr} | "
              f"cloning fails on {r.pct_fail:.0f}% of files")
    print(f"  mean PRR={df.prr.mean():.3f} | mean SNR={df.snr_db.mean():.1f} dB")
    print("=" * 70)


if __name__ == "__main__":
    main()
