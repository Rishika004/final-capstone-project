"""
evaluation/diffpure_experiment.py
===================================
The core experiment for the paper:

  "Does our null-space perturbation SURVIVE real DiffPure?"

Three conditions are compared:
  A) No protection     — baseline (attacker wins trivially)
  B) Random δ          — naive perturbation (no null-space) → DiffPure removes it
  C) Null-space δ      — our method → DiffPure cannot remove it (PRR > 1.0)

Conditions B vs C prove the NULL SPACE is what makes the difference,
not just the size of the perturbation.

Metrics collected for each condition, before and after DiffPure:
  - speaker_cosine_sim : how similar is the speaker embedding to original
  - mel_distance       : how far is the audio from the original mel
  - PRR                : how much the purifier changes the audio

Usage
-----
    python evaluation/diffpure_experiment.py \
        --input LJSpeech-1.1/wavs/ --output results/diffpure/ \
        --n-files 20 --t-diffuse 100 --device cuda

    # Or via main.py (add --mode diffpure-exp)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate    import load_wav, wav_to_logmel, save_wav
from evaluation.metrics     import mel_distance, speaker_cosine_similarity
from core.perturbation      import apply_null_space_protection
from core.nullspace         import project_onto_null_space, compute_null_space_factored
from core.jacobian          import estimate_jacobian_randomized
from models.purifier        import load_pretrained as load_surrogate
from models.real_diffpure   import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_speaker_encoder
from utils.vocoder          import invert_mel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(wav: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
    """Return L2-normalised speaker embedding for a waveform [1, T]."""
    with torch.no_grad():
        emb = encoder(wav)
        return nn.functional.normalize(emb, dim=-1)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())




def _psr(mel_orig_after: torch.Tensor,
         mel_prot_after: torch.Tensor,
         delta: torch.Tensor) -> float:
    """
    Perturbation Survival Rate — the KEY metric for the paper.

    PSR = ||DiffPure(x+δ) - DiffPure(x)||_F / ||δ||_F

    Intuition:
      If DiffPure REMOVES δ:   DiffPure(x+δ) ≈ DiffPure(x)  → PSR ≈ 0
      If DiffPure KEEPS δ:     DiffPure(x+δ) ≈ DiffPure(x)+δ → PSR ≈ 1

    We want PSR_C (null-space) >> PSR_B (random).
    """
    survived = (mel_prot_after.cpu() - mel_orig_after.cpu()).norm().item()
    original  = delta.norm().item() + 1e-12
    return survived / original


# ---------------------------------------------------------------------------
# Single-file experiment
# ---------------------------------------------------------------------------

def run_one_file(
    wav_path: str,
    surrogate_purifier: nn.Module,
    real_diffpure: nn.Module,
    speaker_encoder: nn.Module,
    device: torch.device,
    epsilon: float = 0.01,
    K: int = 50,
    output_dir: str = "results/diffpure/",
    use_real_jacobian: bool = False,
) -> dict:
    """
    Run all three conditions (A/B/C) on one audio file.

    Returns a dict with all metrics for one CSV row.
    """
    filename = Path(wav_path).stem
    print(f"\n{'='*60}")
    print(f"File: {Path(wav_path).name}")

    # ---- Load audio -------------------------------------------------------
    wav_orig = load_wav(wav_path).to(device)
    mel_orig = wav_to_logmel(wav_orig, device)

    # ---- Reference speaker embedding (original, before any attack) --------
    emb_orig = _embed(wav_orig, speaker_encoder)

    # Fixed DiffPure seed — same forward/backward noise for all three conditions.
    # Without this, DiffPure's stochastic noise dominates the tiny delta (ε=0.01)
    # and PSR >> 1.  With a fixed seed, PSR is in [0, 1] and measures only what
    # fraction of delta survives the purification (not noise variance).
    DIFFPURE_SEED = 2026

    # ---- Compute null space ONCE (reused for condition C) ----------------
    if use_real_jacobian:
        # Use the REAL DiffPure (deterministic with DIFFPURE_SEED) to estimate
        # the Jacobian.  The null-space of real DiffPure is what we actually
        # need: perturbations that survive the full 100-step DDPM loop.
        # MelDiffPure.forward() is now differentiable (no torch.no_grad inside),
        # so jvp can propagate tangents through all 100 steps.
        # Cost: ~100x slower than surrogate (~3 min/file vs ~2s/file).
        print("  [1/4] Computing Jacobian + null space (REAL DiffPure) ...")
        jacobian_model = lambda x: real_diffpure(x, seed=DIFFPURE_SEED)
    else:
        print("  [1/4] Computing Jacobian + null space (surrogate) ...")
        jacobian_model = surrogate_purifier

    Z, V = estimate_jacobian_randomized(jacobian_model, mel_orig, K=K, seed=42)
    null_basis, _ = compute_null_space_factored(Z, V, threshold=1e-3)

    # ====================================================================
    # CONDITION A — No protection
    # ====================================================================
    print("  [2/4] Condition A: no protection ...")
    with torch.no_grad():
        mel_A_after  = real_diffpure(mel_orig, seed=DIFFPURE_SEED)
        wav_A_after  = invert_mel(mel_A_after.cpu()).to(device)
        wav_A_after  = _align(wav_A_after, wav_orig)

    emb_A_after  = _embed(wav_A_after, speaker_encoder)
    sim_A_before = _cosine_sim(emb_orig, emb_orig)           # 1.0 by definition
    sim_A_after  = _cosine_sim(emb_orig, emb_A_after)
    # Baseline DiffPure shift (denominator for PRR)
    orig_shift   = (mel_A_after.cpu() - mel_orig.cpu()).norm().item() + 1e-12
    prr_A        = 1.0                                        # by definition

    # ====================================================================
    # CONDITION B — Random δ (no null-space projection)
    # ====================================================================
    print("  [3/4] Condition B: random perturbation (no null space) ...")
    torch.manual_seed(99)
    delta_random = torch.randn_like(mel_orig) * epsilon
    delta_random = delta_random.clamp(-epsilon, epsilon)
    mel_B_before = mel_orig + delta_random

    with torch.no_grad():
        mel_B_after = real_diffpure(mel_B_before, seed=DIFFPURE_SEED)
        wav_B_before = invert_mel(mel_B_before.cpu()).to(device)
        wav_B_after  = invert_mel(mel_B_after.cpu()).to(device)
        wav_B_before = _align(wav_B_before, wav_orig)
        wav_B_after  = _align(wav_B_after,  wav_orig)

    emb_B_before = _embed(wav_B_before, speaker_encoder)
    emb_B_after  = _embed(wav_B_after,  speaker_encoder)
    sim_B_before = _cosine_sim(emb_orig, emb_B_before)
    sim_B_after  = _cosine_sim(emb_orig, emb_B_after)
    # PRR vs real DiffPure: how much does DiffPure shift the protected audio?
    prr_B = (mel_B_after.cpu() - mel_B_before.cpu()).norm().item() / orig_shift
    # PSR: what fraction of delta_B survives DiffPure?
    psr_B = _psr(mel_A_after, mel_B_after, delta_random)

    # ====================================================================
    # CONDITION C — Null-space δ (OUR METHOD)
    # ====================================================================
    print("  [4/4] Condition C: null-space protection (ours) ...")
    mel_C_before, _ = apply_null_space_protection(
        x_mel=mel_orig,
        purifier=surrogate_purifier,
        epsilon=epsilon,
        K=K,
        svd_threshold=1e-3,
        _precomputed_null_basis=null_basis,   # reuse — saves Jacobian time
    )

    with torch.no_grad():
        mel_C_after = real_diffpure(mel_C_before, seed=DIFFPURE_SEED)
        wav_C_before = invert_mel(mel_C_before.cpu()).to(device)
        wav_C_after  = invert_mel(mel_C_after.cpu()).to(device)
        wav_C_before = _align(wav_C_before, wav_orig)
        wav_C_after  = _align(wav_C_after,  wav_orig)

    emb_C_before = _embed(wav_C_before, speaker_encoder)
    emb_C_after  = _embed(wav_C_after,  speaker_encoder)
    sim_C_before = _cosine_sim(emb_orig, emb_C_before)
    sim_C_after  = _cosine_sim(emb_orig, emb_C_after)
    # PRR vs real DiffPure
    prr_C  = (mel_C_after.cpu() - mel_C_before.cpu()).norm().item() / orig_shift
    # PSR: what fraction of delta_C survives DiffPure?
    delta_C = (mel_C_before - mel_orig).cpu()
    psr_C   = _psr(mel_A_after, mel_C_after, delta_C)

    # ---- Save protected wavs for listening --------------------------------
    os.makedirs(output_dir, exist_ok=True)
    save_wav(wav_B_before, os.path.join(output_dir, f"{filename}_B_random_protected.wav"))
    save_wav(wav_B_after,  os.path.join(output_dir, f"{filename}_B_random_after_diffpure.wav"))
    save_wav(wav_C_before, os.path.join(output_dir, f"{filename}_C_nullspace_protected.wav"))
    save_wav(wav_C_after,  os.path.join(output_dir, f"{filename}_C_nullspace_after_diffpure.wav"))

    # ---- Print summary ----------------------------------------------------
    print(f"\n  {'Condition':<25} {'Sim before':<12} {'Sim after':<12} {'PRR(real)':<12} {'PSR':>8}")
    print(f"  {'-'*72}")
    print(f"  {'A: No protection':<25} {sim_A_before:<12.4f} {sim_A_after:<12.4f} {'1.0000':<12} {'—':>8}")
    print(f"  {'B: Random delta':<25} {sim_B_before:<12.4f} {sim_B_after:<12.4f} {prr_B:<12.4f} {psr_B:>8.4f}")
    print(f"  {'C: Null-space (ours)':<25} {sim_C_before:<12.4f} {sim_C_after:<12.4f} {prr_C:<12.4f} {psr_C:>8.4f}")
    print()
    print(f"  KEY METRIC — Perturbation Survival Rate (PSR):")
    print(f"    B (random):     PSR = {psr_B:.4f}  (fraction of random δ that survived DiffPure)")
    print(f"    C (null-space): PSR = {psr_C:.4f}  (fraction of null-space δ that survived DiffPure)")
    psr_lift = psr_C - psr_B
    if psr_lift > 0:
        print(f"    ✓ PSR_C > PSR_B by {psr_lift:.4f} — null-space δ survives better than random")
    else:
        print(f"    ✗ PSR_C <= PSR_B (delta={psr_lift:.4f}) — need stronger purifier or larger epsilon")

    return {
        "filename": Path(wav_path).name,
        "epsilon":  epsilon,
        # Condition A
        "A_sim_before": round(sim_A_before, 4),
        "A_sim_after":  round(sim_A_after,  4),
        "A_prr":        round(prr_A,        4),
        # Condition B — random
        "B_sim_before": round(sim_B_before, 4),
        "B_sim_after":  round(sim_B_after,  4),
        "B_prr":        round(prr_B,        4),
        "B_psr":        round(psr_B,        4),
        # Condition C — ours
        "C_sim_before": round(sim_C_before, 4),
        "C_sim_after":  round(sim_C_after,  4),
        "C_prr":        round(prr_C,        4),
        "C_psr":        round(psr_C,        4),
        # Key deltas
        "B_sim_delta":  round(sim_A_after - sim_B_after, 4),
        "C_sim_delta":  round(sim_A_after - sim_C_after, 4),
        "PSR_lift":     round(psr_C - psr_B, 4),   # positive = null-space survives better
    }


def _align(wav: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Align wav length to reference (Griffin-Lim may produce slightly different T)."""
    orig_len = reference.shape[-1]
    if wav.shape[-1] > orig_len:
        return wav[..., :orig_len]
    elif wav.shape[-1] < orig_len:
        return nn.functional.pad(wav, (0, orig_len - wav.shape[-1]))
    return wav


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_diffpure_experiment(
    audio_dir: str,
    output_dir: str,
    n_files: int = 20,
    epsilon: float = 0.01,
    K: int = 50,
    t_diffuse: int = 100,
    device: torch.device = None,
    use_real_jacobian: bool = False,
) -> pd.DataFrame:
    """
    Run the 3-condition DiffPure experiment on n_files from audio_dir.

    Parameters
    ----------
    audio_dir : str   — directory of .wav files
    output_dir : str  — where to save results + audio
    n_files : int     — how many files to test (20 is enough for the paper)
    epsilon : float   — L-inf perturbation budget
    K : int           — Jacobian probe count
    t_diffuse : int   — DiffPure forward steps (100 = moderate)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"DIFFPURE EXPERIMENT")
    print(f"  audio_dir = {audio_dir}")
    print(f"  n_files   = {n_files}")
    print(f"  epsilon   = {epsilon}")
    print(f"  t_diffuse = {t_diffuse}  (DiffPure forward steps)")
    print(f"  jacobian  = {'real DiffPure' if use_real_jacobian else 'surrogate (fast)'}")
    print(f"  device    = {device}")
    print(f"{'='*60}")

    # Load models
    print("\nLoading models ...")
    surrogate = load_surrogate(device=device)
    real_dp   = load_real_diffpure(t_diffuse=t_diffuse, device=device)
    try:
        encoder = load_speaker_encoder(device=device)
    except Exception as e:
        logger.warning("WavLM unavailable (%s) — using dummy encoder", e)
        from models.speaker_encoder import _make_dummy_encoder
        encoder = _make_dummy_encoder(device)

    # Select files
    wav_files = sorted(Path(audio_dir).glob("*.wav"))[:n_files]
    if not wav_files:
        raise FileNotFoundError(f"No .wav files in {audio_dir}")
    print(f"\nFound {len(wav_files)} files (using first {n_files})")

    # Run per-file
    rows = []
    for wav_path in wav_files:
        try:
            row = run_one_file(
                wav_path=str(wav_path),
                surrogate_purifier=surrogate,
                real_diffpure=real_dp,
                speaker_encoder=encoder,
                device=device,
                epsilon=epsilon,
                K=K,
                output_dir=output_dir,
                use_real_jacobian=use_real_jacobian,
            )
            rows.append(row)
        except Exception as exc:
            logger.error("Failed on %s: %s", wav_path.name, exc, exc_info=True)

    df = pd.DataFrame(rows)

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "diffpure_results.csv")
    df.to_csv(csv_path, index=False)

    # Print summary table
    print(f"\n{'='*72}")
    print(f"DIFFPURE EXPERIMENT — MEAN RESULTS ({len(df)} files, epsilon={epsilon})")
    print(f"{'='*72}")
    print(f"{'Condition':<30} {'Sim before':>10} {'Sim after':>10} {'PRR(real)':>10} {'PSR':>8}")
    print(f"{'-'*72}")
    print(f"{'A: No protection':<30} {df['A_sim_before'].mean():>10.4f} {df['A_sim_after'].mean():>10.4f} {'1.0000':>10} {'—':>8}")
    print(f"{'B: Random delta':<30} {df['B_sim_before'].mean():>10.4f} {df['B_sim_after'].mean():>10.4f} {df['B_prr'].mean():>10.4f} {df['B_psr'].mean():>8.4f}")
    print(f"{'C: Null-space (ours)':<30} {df['C_sim_before'].mean():>10.4f} {df['C_sim_after'].mean():>10.4f} {df['C_prr'].mean():>10.4f} {df['C_psr'].mean():>8.4f}")
    print(f"\n  --- KEY RESULTS ---")
    print(f"  PSR_B (random survives DiffPure):     {df['B_psr'].mean():.4f}  (lower = DiffPure removes it)")
    print(f"  PSR_C (null-space survives DiffPure): {df['C_psr'].mean():.4f}  (higher = DiffPure keeps it)")
    psr_lift = df['PSR_lift'].mean()
    if psr_lift > 0:
        print(f"  PSR lift (C - B): +{psr_lift:.4f}  => null-space delta survives DiffPure better than random")
    else:
        print(f"  PSR lift (C - B): {psr_lift:.4f}  => no survival advantage detected at this epsilon")
    print(f"\n  Speaker sim after DiffPure:")
    print(f"    A (no protection): {df['A_sim_after'].mean():.4f}")
    print(f"    B (random delta):  {df['B_sim_after'].mean():.4f}  (delta vs A: {df['B_sim_delta'].mean():+.4f})")
    print(f"    C (null-space):    {df['C_sim_after'].mean():.4f}  (delta vs A: {df['C_sim_delta'].mean():+.4f})")
    print(f"\nResults saved to: {csv_path}")
    print(f"{'='*72}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="DiffPure experiment: A vs B vs C")
    parser.add_argument("--input",          default="LJSpeech-1.1/wavs/")
    parser.add_argument("--output",         default="results/diffpure/")
    parser.add_argument("--n-files",        type=int,   default=20,   dest="n_files")
    parser.add_argument("--epsilon",        type=float, default=0.01)
    parser.add_argument("--K",              type=int,   default=50)
    parser.add_argument("--t-diffuse",      type=int,   default=100,  dest="t_diffuse")
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--real-jacobian",  action="store_true", dest="real_jacobian",
                        help="Compute null-space Jacobian from real DiffPure (more accurate "
                             "but ~100x slower than surrogate). Expect PSR_C closer to 1.0.")
    args = parser.parse_args()

    dev = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else torch.device(args.device))

    run_diffpure_experiment(
        audio_dir=args.input,
        output_dir=args.output,
        n_files=args.n_files,
        epsilon=args.epsilon,
        K=args.K,
        t_diffuse=args.t_diffuse,
        device=dev,
        use_real_jacobian=args.real_jacobian,
    )
