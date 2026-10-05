"""
evaluation/evaluate.py
======================
Full evaluation pipeline for null-space voice protection.

For each .wav file in an input directory the pipeline:
  1. Loads the audio and converts to log-mel spectrogram.
  2. Applies null-space protection.
  3. Applies the surrogate purifier to both original and protected audio.
  4. Computes all evaluation metrics (PRR, cosine similarity, PESQ, SNR, mel dist).
  5. Saves the protected waveform as a .wav file.
  6. Aggregates results into a pandas DataFrame and saves results.csv.

Usage (programmatic)
--------------------
    from evaluation.evaluate import run_full_evaluation
    df = run_full_evaluation("data/samples/", "results/")

Usage (CLI)
-----------
    python main.py --mode evaluate --input data/samples/ --output results/
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
try:
    import torchaudio.transforms as T
except Exception:   # Smart App Control blocks libtorchaudio.pyd (2026-09-15)
    from utils import tacompat as T
import pandas as pd

from core.perturbation import apply_null_space_protection
from evaluation.metrics import compute_all_metrics
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_speaker_encoder
from utils.audio_io import load_wav as _load_wav_impl, save_wav as _save_wav_impl
from utils.vocoder import invert_mel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio processing constants (must match training config of all models)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
MAX_AUDIO_SECONDS = 10  # clip long files to avoid OOM


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def load_wav(
    path: str,
    target_sr: int = SAMPLE_RATE,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> torch.Tensor:
    """
    Load a .wav file and return a mono 16kHz waveform tensor [1, T].
    Uses soundfile backend (no FFmpeg required).
    """
    wav, _ = _load_wav_impl(path, target_sr=target_sr, max_seconds=max_seconds)
    return wav


def wav_to_logmel(wav: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert [1, T] waveform → [1, N_MELS, time] log-mel spectrogram."""
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    ).to(device)
    mel = mel_transform(wav.to(device))
    return torch.log(mel + 1e-9)


def save_wav(wav: torch.Tensor, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a [1, T] waveform tensor to a .wav file (uses soundfile, no FFmpeg)."""
    _save_wav_impl(wav, path, sample_rate)


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_single_file(
    wav_path: str,
    output_dir: str,
    purifier: torch.nn.Module,
    speaker_encoder: torch.nn.Module,
    device: torch.device,
    epsilon: float = 0.01,
    K: int = 50,
    svd_threshold: float = 1e-3,
    mode: str = "random",
    pgd_steps: int = 50,
    perceptual: bool = True,
    masking_margin_db: float = 25.0,
) -> dict:
    """
    Run the full protection + evaluation pipeline for one audio file.

    Parameters
    ----------
    wav_path : str
        Path to input .wav file.
    output_dir : str
        Directory where protected audio will be saved.
    purifier, speaker_encoder : nn.Module
        Pretrained models.
    device : torch.device
    epsilon : float
        L∞ perturbation budget.
    K : int
        Jacobian probe count.
    svd_threshold : float
        SVD threshold for null-space extraction.

    Returns
    -------
    row : dict
        Evaluation results for this file (one row of the results DataFrame).
    """
    filename = Path(wav_path).stem
    logger.info("Processing: %s", wav_path)
    print(f"\n{'='*60}")
    print(f"[Evaluate] File: {Path(wav_path).name}")

    t_start = time.perf_counter()

    # ---------------------------------------------------------------
    # 1. Load audio
    # ---------------------------------------------------------------
    wav_original = load_wav(wav_path).to(device)
    mel_original = wav_to_logmel(wav_original, device)
    print(f"  Audio loaded: {wav_original.shape[-1] / SAMPLE_RATE:.2f}s | "
          f"mel shape: {list(mel_original.shape)}")

    # ---------------------------------------------------------------
    # 2. Apply null-space protection (returns protected mel)
    # ---------------------------------------------------------------
    print(f"  Applying null-space protection (mode={mode}) ...")
    mel_protected, best_wav_delta = apply_null_space_protection(
        x_mel=mel_original,
        purifier=purifier,
        epsilon=epsilon,
        K=K,
        svd_threshold=svd_threshold,
        verify=True,
        mode=mode,
        x_wav=wav_original if mode == "pgd" else None,
        speaker_encoder=speaker_encoder if mode == "pgd" else None,
        pgd_steps=pgd_steps,
        perceptual=perceptual,
        masking_margin_db=masking_margin_db,
    )

    # ---------------------------------------------------------------
    # 3. Convert protected mel back to waveform.
    #
    # PGD mode  → bypass Griffin-Lim: add the raw Stage-1 waveform delta
    #   directly to the original waveform.  Stage 1 PGD achieved
    #   cos_sim ~ -0.47 in waveform space; Griffin-Lim destroyed that
    #   adversarial signal because it reinitialises phase randomly.
    #   Using wav_original + best_wav_delta preserves it perfectly.
    #
    # Random mode → Griffin-Lim (no waveform delta available).
    # ---------------------------------------------------------------
    orig_len = wav_original.shape[-1]
    with torch.no_grad():
        if best_wav_delta is not None:
            print("  Building protected waveform via direct waveform delta (bypassing Griffin-Lim) ...")
            wav_delta_dev = best_wav_delta.to(device)
            # Align length in case of any shape mismatch
            dlen = wav_delta_dev.shape[-1]
            if dlen > orig_len:
                wav_delta_dev = wav_delta_dev[..., :orig_len]
            elif dlen < orig_len:
                wav_delta_dev = torch.nn.functional.pad(wav_delta_dev, (0, orig_len - dlen))
            wav_protected = (wav_original + wav_delta_dev).clamp(-1.0, 1.0)
            print(f"  wav_delta L_inf = {wav_delta_dev.abs().max().item():.5f}")
        else:
            print("  Inverting protected mel to waveform via Griffin-Lim ...")
            wav_protected = invert_mel(mel_protected.cpu(), method="griffin_lim")
            wav_protected = wav_protected.to(device)
            # Griffin-Lim may produce slightly different length than original
            prot_len = wav_protected.shape[-1]
            if prot_len > orig_len:
                wav_protected = wav_protected[..., :orig_len]
            elif prot_len < orig_len:
                wav_protected = torch.nn.functional.pad(wav_protected, (0, orig_len - prot_len))

    # ---------------------------------------------------------------
    # 4. UNIFIED MEASURED SIGNAL (review finding #1)
    #
    # Every metric below is computed on the ONE waveform we actually save
    # (wav_protected). We re-derive its log-mel here so that survival (PSR),
    # speaker similarity and quality all describe the SAME signal, instead of
    # measuring survival on a separately-null-projected mel that the delivered
    # audio never contained.
    #
    # mel_protected (the null-projected mel returned by the protection) is kept
    # only to report the theoretical-vs-delivered survival GAP, so the effect of
    # bypassing the null projection in the waveform is visible rather than hidden.
    # ---------------------------------------------------------------
    mel_delivered = wav_to_logmel(wav_protected, device)

    print("  Applying purifier to original and delivered ...")
    with torch.no_grad():
        mel_orig_purified = purifier(mel_original)
        mel_prot_purified = purifier(mel_delivered)

    # ---------------------------------------------------------------
    # 5. Compute metrics — all on the delivered signal
    # ---------------------------------------------------------------
    print("  Computing metrics ...")
    metrics = compute_all_metrics(
        wav_original=wav_original,
        wav_protected=wav_protected,
        mel_original=mel_original,
        mel_protected=mel_delivered,   # <- delivered mel, not the decoupled one
        purifier=purifier,
        speaker_encoder=speaker_encoder,
        sample_rate=SAMPLE_RATE,
    )

    # Additional metrics
    with torch.no_grad():
        from evaluation.metrics import (mel_distance, speaker_cosine_similarity,
                                        signal_to_noise_ratio,
                                        perturbation_survival_rate)
        metrics["mel_dist_after_purify"] = mel_distance(
            mel_orig_purified, mel_prot_purified
        )
        # Survival GAP: PSR of the theoretical null-projected mel vs. the mel we
        # actually deliver. A large gap means the null-space guarantee does not
        # transfer to the shipped waveform (the honest number is psr, on delivered).
        metrics["psr_projected"] = perturbation_survival_rate(
            mel_original, mel_protected, purifier
        )
        metrics["psr_delivered"] = metrics["psr"]
        # SNR computed in mel space — avoids Griffin-Lim phase reconstruction artifacts.
        # Mel-SNR is the true measure of how imperceptible our perturbation is.
        mel_signal_power = mel_original.pow(2).mean().item()
        mel_noise_power  = (mel_protected - mel_original).pow(2).mean().item()
        import math
        metrics["snr_db"] = 10.0 * math.log10(mel_signal_power / (mel_noise_power + 1e-20))

        # Waveform-domain SNR + silence noise level — the honest audibility
        # numbers (mel-space SNR flatters the perturbation considerably).
        from core.psychoacoustic import waveform_snr_db, silence_noise_db
        metrics["snr_wav_db"] = waveform_snr_db(wav_original.cpu(), wav_protected.cpu())
        sil_db, n_sil = silence_noise_db(wav_original.cpu(), wav_protected.cpu(),
                                         sample_rate=SAMPLE_RATE)
        metrics["silence_noise_db"] = sil_db
        metrics["n_silent_frames"] = n_sil

    elapsed = time.perf_counter() - t_start

    # ---------------------------------------------------------------
    # 6. Save protected audio
    # ---------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename}_protected.wav")
    save_wav(wav_protected, out_path)

    # Rename speaker_cosine_sim → speaker_similarity for consistent CSV column
    if "speaker_cosine_sim" in metrics:
        metrics["speaker_similarity"] = metrics.pop("speaker_cosine_sim")

    row = {
        "filename": Path(wav_path).name,
        "duration_s": round(wav_original.shape[-1] / SAMPLE_RATE, 2),
        "epsilon": epsilon,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "processing_time_s": round(elapsed, 2),
        "output_path": out_path,
    }

    print(f"  PSR (delivered)   = {metrics['psr']:.4f}  (~1.0 = δ survives, ~0 = removed)")
    print(f"  PSR (projected)   = {metrics['psr_projected']:.4f}  (theoretical null-proj mel)")
    print(f"  PRR (legacy)      = {metrics['prr']:.4f}  (=1.0 at δ=0; not a survival proof)")
    print(f"  Speaker cos-sim   = {metrics['speaker_similarity']:.4f}  (<0.5 is good)")
    print(f"  Mel distance      = {metrics['mel_distance']:.6f}")
    print(f"  PESQ score        = {metrics.get('pesq', float('nan')):.3f}  (>3.0 is good)")
    print(f"  STOI score        = {metrics.get('stoi', float('nan')):.4f}  (>0.95 is good)")
    print(f"  SNR wav (dB)      = {metrics.get('snr_wav_db', float('nan')):.2f}  (honest, waveform)")
    print(f"  Elapsed           = {elapsed:.2f}s")
    print(f"  Saved to          : {out_path}")

    return row


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    audio_dir: str,
    output_dir: str,
    epsilon: float = 0.01,
    K: int = 50,
    svd_threshold: float = 1e-3,
    device: Optional[torch.device] = None,
    mode: str = "random",
    pgd_steps: int = 50,
    n_files: Optional[int] = None,
    perceptual: bool = True,
    masking_margin_db: float = 25.0,
) -> pd.DataFrame:
    """
    Evaluate null-space protection on all .wav files in audio_dir.

    Parameters
    ----------
    audio_dir : str
        Directory containing .wav input files.
    output_dir : str
        Directory to write protected audio and results.csv.
    epsilon : float
        L∞ perturbation budget.
    K : int
        Jacobian probe count.
    svd_threshold : float
        SVD threshold for null-space extraction.
    device : torch.device, optional

    Returns
    -------
    df : pd.DataFrame
        One row per file, columns = all metric names.
        Also saved to {output_dir}/results.csv.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")

    # Find all wav files
    audio_path = Path(audio_dir)
    wav_files = sorted(audio_path.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in '{audio_dir}'. "
            "Put some .wav files there and try again."
        )
    if n_files is not None:
        wav_files = wav_files[:n_files]
    print(f"[Evaluate] Found {len(wav_files)} .wav file(s) in {audio_dir}")

    # Load models once (reused across all files)
    print("\n[Evaluate] Loading models ...")
    # Use real DiffPure (DDPM loop) as the purifier.
    # t_diffuse=50 gives a good balance: strong enough to test real
    # purification resistance, fast enough for practical evaluation.
    purifier = load_real_diffpure(t_diffuse=50, device=device)
    try:
        speaker_encoder = load_speaker_encoder(device=device)
    except Exception as exc:
        logger.warning("Could not load WavLM speaker encoder: %s", exc)
        from models.speaker_encoder import _make_dummy_encoder
        speaker_encoder = _make_dummy_encoder(device)

    # Process each file
    results = []
    for wav_path in wav_files:
        try:
            row = evaluate_single_file(
                wav_path=str(wav_path),
                output_dir=output_dir,
                purifier=purifier,
                speaker_encoder=speaker_encoder,
                device=device,
                epsilon=epsilon,
                K=K,
                svd_threshold=svd_threshold,
                mode=mode,
                pgd_steps=pgd_steps,
                perceptual=perceptual,
                masking_margin_db=masking_margin_db,
            )
            results.append(row)
        except Exception as exc:
            logger.error("Failed to process %s: %s", wav_path.name, exc, exc_info=True)
            results.append({"filename": wav_path.name, "error": str(exc)})

    # Build DataFrame
    df = pd.DataFrame(results)

    # Save results.csv
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Evaluate] Results saved to {csv_path}")

    # Print summary table
    numeric_cols = ["psr", "psr_projected", "prr", "speaker_similarity",
                    "mel_distance", "pesq", "stoi", "snr_wav_db", "snr_db"]
    summary_cols = [c for c in numeric_cols if c in df.columns]
    if summary_cols:
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(df[["filename"] + summary_cols].to_string(index=False))
        print("-"*70)
        print("MEAN VALUES:")
        for col in summary_cols:
            try:
                mean_val = df[col].astype(float).mean()
                print(f"  {col:30s} = {mean_val:.4f}")
            except Exception:
                pass
        print("="*70)

    return df
