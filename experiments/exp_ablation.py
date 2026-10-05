"""
experiments/exp_ablation.py
===========================
Hyperparameter sensitivity (ablation) study.

Baseline config:  K=50, epsilon=0.01, pgd_steps=50, svd_threshold=1e-3
We vary ONE parameter at a time and record all metrics.

Efficiency trick
----------------
The Jacobian probes are drawn sequentially from a seeded RNG, so the first
K columns of a K=100 estimate are EXACTLY the estimate a K-probe run would
produce. We therefore compute the Jacobian ONCE at K=100 per file and slice
[:, :K] for every smaller K. The PGD attack (fast on GPU) is re-run only for
the epsilon and pgd_steps sweeps.

Sweeps
------
  K              in {10, 25, 50, 100}
  svd_threshold  in {1e-4, 1e-3, 1e-2}
  epsilon        in {0.005, 0.01, 0.02}
  pgd_steps      in {10, 25, 50}

Metrics
-------
  null_dim, cos_sim (WavLM), range_energy_frac, prr, snr_db, mel_dist

Run:
  python experiments/exp_ablation.py --n-files 1
Output:
  experiments/results/ablation.csv
"""

import sys, io, os, argparse, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from core.jacobian import estimate_jacobian_randomized
from core.nullspace import compute_null_space_factored, project_onto_null_space
from core.perturbation import generate_pgd_perturbation
from evaluation.evaluate import load_wav, wav_to_logmel
from evaluation.metrics import purification_resistance_rate, speaker_cosine_similarity
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_speaker_encoder

OUT_DIR = os.path.join(PROJ, "experiments", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BASE = dict(K=50, epsilon=0.01, pgd_steps=50, threshold=1e-3)
K_MAX = 100


def range_energy_frac(delta_mel, range_basis):
    d = delta_mel.reshape(-1).float().cpu()
    nb = range_basis.float().cpu()
    if nb.shape[0] == 0:
        return 0.0
    coords = nb @ d
    return coords.pow(2).sum().item() / (d.pow(2).sum().item() + 1e-20)


def snr_mel(mel_o, mel_p):
    s = mel_o.pow(2).mean().item()
    n = (mel_p - mel_o).pow(2).mean().item()
    return 10.0 * math.log10(s / (n + 1e-20))


def wav_from_delta(wav_o, wav_delta):
    d = wav_delta.to(wav_o.device)
    L = wav_o.shape[-1]
    if d.shape[-1] > L:
        d = d[..., :L]
    elif d.shape[-1] < L:
        d = torch.nn.functional.pad(d, (0, L - d.shape[-1]))
    return (wav_o + d).clamp(-1.0, 1.0)


def evaluate_config(mel_o, wav_o, purifier, speaker_encoder, device,
                    Z_full, V_full, K, epsilon, pgd_steps, threshold,
                    pgd_cache):
    """Return a metrics dict for one config. pgd_cache keyed by (epsilon,pgd_steps)."""
    # Null space from sliced Jacobian
    Z = Z_full[:, :K]
    V = V_full[:, :K]
    range_basis, S = compute_null_space_factored(Z, V, threshold=threshold)
    null_dim = mel_o.numel() - range_basis.shape[0]

    # PGD delta (cache by eps+steps; cos-sim independent of null basis)
    key = (round(epsilon, 5), pgd_steps)
    if key not in pgd_cache:
        pgd_mel, pgd_wav_delta = generate_pgd_perturbation(
            x_mel=mel_o, x_wav=wav_o, speaker_encoder=speaker_encoder,
            epsilon=epsilon, n_steps=pgd_steps, seed=42)
        wav_p = wav_from_delta(wav_o, pgd_wav_delta)
        cos = speaker_cosine_similarity(wav_o, wav_p, speaker_encoder)
        pgd_cache[key] = (pgd_mel, cos)
    pgd_mel, cos = pgd_cache[key]

    # Project onto null space, clamp
    pgd_null = project_onto_null_space(pgd_mel, range_basis).clamp(-epsilon, epsilon).to(device)
    prr = purification_resistance_rate(mel_o, mel_o + pgd_null, purifier)

    return dict(
        null_dim=null_dim,
        cos_sim=round(cos, 4),
        range_energy_frac=round(range_energy_frac(pgd_null.cpu(), range_basis), 5),
        prr=round(prr, 4),
        snr_db=round(snr_mel(mel_o, mel_o + pgd_null), 2),
        mel_dist=round(pgd_null.pow(2).mean().sqrt().item(), 6),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--n-files", type=int, default=1, dest="n_files")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ablation] device={device}")

    wavs = sorted(Path(args.input).glob("*.wav"))[:args.n_files]
    purifier = load_real_diffpure(t_diffuse=50, device=device)
    speaker_encoder = load_speaker_encoder(device=device)

    rows = []
    for wp in wavs:
        print(f"\n{'='*60}\n[Ablation] {wp.name}")
        wav_o = load_wav(str(wp)).to(device)
        mel_o = wav_to_logmel(wav_o, device)

        print(f"  Jacobian once at K={K_MAX} (reused via slicing) ...")
        Z_full, V_full = estimate_jacobian_randomized(purifier, mel_o, K=K_MAX, seed=42)
        pgd_cache = {}

        def run(tag, **cfg):
            c = {**BASE, **cfg}
            m = evaluate_config(mel_o, wav_o, purifier, speaker_encoder, device,
                                Z_full, V_full, c["K"], c["epsilon"],
                                c["pgd_steps"], c["threshold"], pgd_cache)
            row = dict(file=wp.name, sweep=tag,
                       K=c["K"], epsilon=c["epsilon"],
                       pgd_steps=c["pgd_steps"], threshold=c["threshold"], **m)
            rows.append(row)
            print(f"  [{tag:9s}] K={c['K']:>3} eps={c['epsilon']:<5} steps={c['pgd_steps']:>2} "
                  f"thr={c['threshold']:<6} | null_dim={m['null_dim']:>5} "
                  f"cos={m['cos_sim']:+.3f} range_frac={m['range_energy_frac']:.4f} "
                  f"PRR={m['prr']:.3f} SNR={m['snr_db']:.1f}")

        # K sweep
        for K in [10, 25, 50, 100]:
            run("K", K=K)
        # threshold sweep
        for thr in [1e-4, 1e-3, 1e-2]:
            run("threshold", threshold=thr)
        # epsilon sweep
        for eps in [0.005, 0.01, 0.02]:
            run("epsilon", epsilon=eps)
        # pgd_steps sweep
        for st in [10, 25, 50]:
            run("pgd_steps", pgd_steps=st)

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "ablation.csv")
    df.to_csv(csv, index=False)
    print(f"\n[Ablation] saved {csv}")

    # Per-sweep summary (mean across files)
    for sweep, param in [("K", "K"), ("threshold", "threshold"),
                         ("epsilon", "epsilon"), ("pgd_steps", "pgd_steps")]:
        sub = df[df.sweep == sweep]
        print(f"\n=== {sweep} sweep (mean across files) ===")
        g = sub.groupby(param).agg(
            null_dim=("null_dim", "mean"), cos=("cos_sim", "mean"),
            range_frac=("range_energy_frac", "mean"), prr=("prr", "mean"),
            snr=("snr_db", "mean"))
        for val, r in g.iterrows():
            print(f"  {param}={val:<7} | null_dim={r.null_dim:>7.0f} cos={r.cos:+.3f} "
                  f"range_frac={r.range_frac:.4f} PRR={r.prr:.3f} SNR={r.snr:.1f}")


if __name__ == "__main__":
    main()
