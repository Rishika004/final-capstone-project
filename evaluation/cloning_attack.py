"""
evaluation/cloning_attack.py
============================
Evaluate voice-cloning resistance of the null-space protection.

Attack model
------------
Voice cloning systems (YourTTS, SV2TTS, XTTS-v2) operate in two stages:
  1. **Speaker conditioning**: Extract a d-dimensional speaker embedding
     from a short reference clip.
  2. **Synthesis**: Condition a TTS model on that embedding and generate
     speech that mimics the speaker's voice.

Our defence succeeds when the embedding extracted from the PROTECTED audio
diverges far enough from the original that the synthesiser generates speech
for a *different* identity.

Metrics
-------
Cloning Success Rate (CSR)
    CSR = cosine_sim(embed(clone(x_protected)), embed(x_original))
    Lower is better.  Threshold = 0.75 (standard speaker-verification threshold).

Defence Effectiveness (DE)
    DE = 1 − CSR / baseline_CSR
    0 = no protection, 1 = perfect de-identification.

Two evaluation modes
--------------------
1. **Embedding-level simulation** (fast, no extra downloads)
   Directly measures cosine similarity of the *input* embeddings.
   This is exactly the conditioning signal that cloning uses.
   Run time: < 1 s per file.

2. **YourTTS actual cloning** (best accuracy, requires Coqui TTS)
   pip install TTS
   Synthesises speech using protected audio as reference, then measures
   speaker similarity of the output clone.
   Run time: ~30–60 s per file (CPU).

Usage
-----
    # Embedding simulation (default)
    python main.py --mode clone-eval --input data/samples/ --output results/

    # Real YourTTS attack
    python main.py --mode clone-eval --input data/samples/ --output results/ --use-yourtts
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Cosine-similarity threshold above which a clone counts as "successful".
#
# 0.75 WAS HARDCODED HERE AND IT MATCHES NOTHING. It was commented as "standard
# speaker-verification threshold", but a verified sweep of the six baseline papers
# (2026-09-16, see paper/TODO_BEFORE_SUBMISSION.md B3) found no paper using it, and
# it contradicts this project's own EER-calibrated thresholds (ECAPA 0.29-0.49
# depending on channel). A wrong threshold here does not crash -- it silently emits
# a full table of meaningless protection rates.
#
# Two defensible conventions, and we report BOTH:
#   EER-calibrated per channel  -- this project's practice, the stricter one.
#       Source it from experiments/calibrate_thresholds.py through the SAME channel
#       the clone passed through. Reusing a threshold across channels is invalid.
#   0.25 on ECAPA-TDNN          -- the field convention, used by SafeSpeech,
#       VoiceCloak and ClearMask (De-AntiFake uses 0.25 on x-vector). Originates in
#       Jiang et al., "Mega-TTS 2", ICLR 2024. Report it for comparability.
#
# There is deliberately no module-level default: callers must pass a threshold, so
# that a missing one is a TypeError rather than a plausible-looking wrong number.
FIELD_CONVENTION_THRESHOLD = 0.25   # ECAPA-TDNN cosine; for comparability only


def _require_threshold(threshold):
    if threshold is None:
        raise ValueError(
            "cloning_attack: a threshold must be supplied explicitly. Use the "
            "EER-calibrated value for the channel this clone passed through (see "
            "experiments/calibrate_thresholds.py), and additionally report at "
            "FIELD_CONVENTION_THRESHOLD=0.25 on ECAPA for comparability with "
            "SafeSpeech / VoiceCloak / ClearMask / De-AntiFake.")
    return float(threshold)


# ---------------------------------------------------------------------------
# Helper: cosine similarity between two embedding tensors
# ---------------------------------------------------------------------------

def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float().reshape(1, -1), dim=-1)
    b = F.normalize(b.float().reshape(1, -1), dim=-1)
    return (a * b).sum().item()


# ---------------------------------------------------------------------------
# Mode 1: Embedding-level simulation attack
# ---------------------------------------------------------------------------

def embedding_similarity_attack(
    wav_original: torch.Tensor,
    wav_protected: torch.Tensor,
    speaker_encoder: torch.nn.Module,
    threshold: float = None,   # REQUIRED; see _require_threshold
) -> Dict[str, object]:
    """
    Simulate a voice-cloning attack via speaker embeddings.

    The speaker conditioning signal of every voice cloning system is the
    speaker embedding.  Measuring the cosine distance between the original
    and protected embeddings therefore directly quantifies how much the
    cloning system's conditioning has been disrupted.

    Parameters
    ----------
    wav_original  : torch.Tensor [1, T]
    wav_protected : torch.Tensor [1, T]
    speaker_encoder : nn.Module

    Returns
    -------
    dict with keys:
        emb_sim_orig_prot   — cosine similarity original vs protected
        cloning_blocked     — True if sim < `threshold` (which is required)
        defence_effectiveness — float in [0, 1]
        attack_mode         — "embedding_simulation"
    """
    speaker_encoder.eval()
    with torch.no_grad():
        emb_orig = speaker_encoder(wav_original)
        emb_prot = speaker_encoder(wav_protected)

    sim = _cosine(emb_orig, emb_prot)
    tau = _require_threshold(threshold)
    cloning_blocked      = sim < tau
    defence_effectiveness = max(0.0, 1.0 - sim)   # range [0,1] since sim ∈ [-1,1]

    result = {
        "emb_sim_orig_prot":     round(sim, 4),
        "cloning_blocked":       cloning_blocked,
        "clone_success_threshold": tau,
        "blocked_at_field_convention": sim < FIELD_CONVENTION_THRESHOLD,
        "defence_effectiveness": round(defence_effectiveness, 4),
        "attack_mode":           "embedding_simulation",
    }
    logger.info("Embedding attack: sim=%.4f | blocked=%s | DE=%.4f",
                sim, cloning_blocked, defence_effectiveness)
    return result


# ---------------------------------------------------------------------------
# Mode 2: YourTTS actual cloning attack
# ---------------------------------------------------------------------------

_yourtts_model = None   # module-level cache


def _load_yourtts(device: torch.device):
    """Load YourTTS from Coqui TTS, cached after first call."""
    global _yourtts_model
    if _yourtts_model is not None:
        return _yourtts_model
    try:
        from TTS.api import TTS
        print("[CloningAttack] Loading YourTTS — may download ~1.5 GB on first run ...")
        _yourtts_model = TTS(
            model_name="tts_models/multilingual/multi-dataset/your_tts",
            progress_bar=True,
        )
        print("[CloningAttack] YourTTS ready.")
        return _yourtts_model
    except ImportError:
        logger.warning("Coqui TTS not installed.  Run:  pip install TTS")
        return None
    except Exception as exc:
        logger.warning("YourTTS load failed: %s", exc)
        return None


def yourtts_cloning_attack(
    wav_original: torch.Tensor,
    wav_protected: torch.Tensor,
    speaker_encoder: torch.nn.Module,
    threshold: float = None,   # REQUIRED; see _require_threshold
    text: str = "This is a test of the voice cloning system.",
    sample_rate: int = 16000,
    tmp_dir: str = "tmp_cloning",
    device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    """
    Run a real YourTTS voice-cloning attack and measure success.

    Pipeline
    --------
    1. Save original and protected audio to temporary WAV files.
    2. Clone voice using YourTTS with original reference → clone_from_orig.
    3. Clone voice using YourTTS with protected reference → clone_from_prot.
    4. Measure speaker similarity:
       - sim(clone_from_orig,  original)  → baseline cloning quality (should be high)
       - sim(clone_from_prot,  original)  → attack on protected (should be low)
    5. Report whether cloning succeeded.

    Falls back to embedding_similarity_attack if YourTTS is unavailable.

    Returns
    -------
    dict with keys:
        clone_orig_sim       — sim(clone_from_original, original)
        clone_prot_sim       — sim(clone_from_protected, original)  [KEY]
        clone_degradation    — clone_orig_sim - clone_prot_sim
        cloning_blocked      — True if clone_prot_sim < threshold
        defence_effectiveness
        attack_mode          — "yourtts"
    """
    import soundfile as sf

    tts = _load_yourtts(device)
    if tts is None:
        logger.info("Falling back to embedding-similarity attack.")
        return embedding_similarity_attack(wav_original, wav_protected,
                                           speaker_encoder, threshold=threshold)

    os.makedirs(tmp_dir, exist_ok=True)
    orig_ref_path     = os.path.join(tmp_dir, "ref_original.wav")
    prot_ref_path     = os.path.join(tmp_dir, "ref_protected.wav")
    clone_orig_path   = os.path.join(tmp_dir, "clone_from_original.wav")
    clone_prot_path   = os.path.join(tmp_dir, "clone_from_protected.wav")

    # Save reference clips
    sf.write(orig_ref_path, wav_original.detach().cpu().squeeze().numpy(), sample_rate)
    sf.write(prot_ref_path, wav_protected.detach().cpu().squeeze().numpy(), sample_rate)

    # --- Synthesis --------------------------------------------------------
    for ref_path, out_path, label in [
        (orig_ref_path, clone_orig_path, "ORIGINAL"),
        (prot_ref_path, clone_prot_path, "PROTECTED"),
    ]:
        try:
            print(f"[CloningAttack] Synthesising from {label} reference ...")
            tts.tts_to_file(
                text        = text,
                speaker_wav = ref_path,
                language    = "en",
                file_path   = out_path,
            )
        except Exception as exc:
            logger.warning("YourTTS synthesis from %s failed: %s", label, exc)
            return embedding_similarity_attack(wav_original, wav_protected,
                                           speaker_encoder, threshold=threshold)

    # --- Load synthesised clips & embed -----------------------------------
    from utils.audio_io import load_wav
    try:
        clone_orig_wav, _ = load_wav(clone_orig_path, target_sr=sample_rate)
        clone_prot_wav, _ = load_wav(clone_prot_path, target_sr=sample_rate)
    except Exception as exc:
        logger.warning("Could not reload synthesised audio: %s", exc)
        return embedding_similarity_attack(wav_original, wav_protected,
                                           speaker_encoder, threshold=threshold)

    speaker_encoder.eval()
    with torch.no_grad():
        emb_orig      = speaker_encoder(wav_original.to(device))
        emb_clone_o   = speaker_encoder(clone_orig_wav.to(device))
        emb_clone_p   = speaker_encoder(clone_prot_wav.to(device))

    sim_clone_orig = _cosine(emb_clone_o, emb_orig)  # baseline clone quality
    sim_clone_prot = _cosine(emb_clone_p, emb_orig)  # attack quality on protected

    tau = _require_threshold(threshold)
    cloning_blocked      = sim_clone_prot < tau
    clone_degradation    = max(0.0, sim_clone_orig - sim_clone_prot)
    defence_effectiveness = clone_degradation / max(sim_clone_orig, 1e-6)

    result = {
        "clone_orig_sim":         round(sim_clone_orig, 4),
        "clone_prot_sim":         round(sim_clone_prot, 4),
        "clone_degradation":      round(clone_degradation, 4),
        "cloning_blocked":        cloning_blocked,
        "clone_success_threshold": tau,
        "blocked_at_field_convention": sim_clone_prot < FIELD_CONVENTION_THRESHOLD,
        "defence_effectiveness":  round(defence_effectiveness, 4),
        "attack_mode":            "yourtts",
    }
    logger.info("YourTTS attack: baseline=%.4f | protected=%.4f | blocked=%s",
                sim_clone_orig, sim_clone_prot, cloning_blocked)
    print(
        f"\n[CloningAttack] YourTTS results:\n"
        f"  Clone(original)  vs speaker: {sim_clone_orig:.4f}\n"
        f"  Clone(protected) vs speaker: {sim_clone_prot:.4f}  "
        f"({'BLOCKED' if cloning_blocked else 'SUCCEEDED'})\n"
        f"  Clone degradation          : {clone_degradation:.4f}\n"
        f"  Defence effectiveness      : {defence_effectiveness:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Batch evaluation over a directory
# ---------------------------------------------------------------------------

def run_cloning_evaluation(
    audio_dir: str,
    output_dir: str,
    purifier: torch.nn.Module,
    speaker_encoder: torch.nn.Module,
    epsilon: float = 0.01,
    K: int = 50,
    svd_threshold: float = 1e-3,
    use_yourtts: bool = False,
    device: torch.device = torch.device("cpu"),
    text: str = "Hello, this is a test of the voice protection system.",
) -> List[Dict]:
    """
    Evaluate cloning resistance for all .wav files in audio_dir.

    Parameters
    ----------
    use_yourtts : bool
        If True, use real YourTTS synthesis (requires pip install TTS).
        Default is the faster embedding-similarity simulation.

    Returns
    -------
    results : list of dicts (one per file), also saved to cloning_results.csv.
    """
    import torchaudio.transforms as T
    import pandas as pd
    from core.perturbation import apply_null_space_protection
    from utils.audio_io import load_wav
    from utils.vocoder import invert_mel

    SAMPLE_RATE = 16000
    N_MELS, N_FFT, HOP_LENGTH = 80, 1024, 256

    mel_fn = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    ).to(device)

    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files in '{audio_dir}'")

    purifier.eval()
    speaker_encoder.eval()
    all_results = []

    print(f"\n[CloningEval] Evaluating {len(wav_files)} file(s) | "
          f"mode={'yourtts' if use_yourtts else 'embedding_simulation'}")

    for wav_path in wav_files:
        print(f"\n[CloningEval] {'='*50}")
        print(f"[CloningEval] File: {wav_path.name}")
        try:
            # Load audio
            wav, _ = load_wav(str(wav_path), target_sr=SAMPLE_RATE)
            wav = wav.to(device)

            # Mel conversion
            mel = torch.log(mel_fn(wav) + 1e-9)

            # Apply null-space protection
            mel_protected, _ = apply_null_space_protection(
                x_mel          = mel,
                purifier       = purifier,
                epsilon        = epsilon,
                K              = K,
                svd_threshold  = svd_threshold,
                verify         = False,
            )

            # Invert protected mel → waveform with Griffin-Lim
            wav_protected = invert_mel(mel_protected.cpu(), method="griffin_lim")
            wav_protected = wav_protected.to(device)

            # Run cloning attack
            if use_yourtts:
                result = yourtts_cloning_attack(
                    wav_original  = wav,
                    wav_protected = wav_protected,
                    speaker_encoder = speaker_encoder,
                    text          = text,
                    device        = device,
                    tmp_dir       = os.path.join(output_dir, "tmp_cloning"),
                )
            else:
                result = embedding_similarity_attack(wav, wav_protected, speaker_encoder)

            result["filename"] = wav_path.name
            result["epsilon"]  = epsilon
            all_results.append(result)

            de  = result.get("defence_effectiveness", float("nan"))
            blk = result.get("cloning_blocked", "?")
            print(f"  Defence effectiveness : {de:.4f}")
            print(f"  Cloning blocked       : {blk}")

        except Exception as exc:
            logger.error("Failed %s: %s", wav_path.name, exc, exc_info=True)
            all_results.append({"filename": wav_path.name, "error": str(exc)})

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLONING RESISTANCE SUMMARY")
    print("=" * 60)
    valid   = [r for r in all_results if "error" not in r]
    blocked = sum(1 for r in valid if r.get("cloning_blocked", False))
    total   = len(valid)
    print(f"  Files evaluated         : {total}")
    print(f"  Cloning blocked         : {blocked}/{total}  "
          f"({100 * blocked / max(total, 1):.1f}%)")

    sims = [r["emb_sim_orig_prot"] for r in valid if "emb_sim_orig_prot" in r]
    if sims:
        print(f"  Mean embedding sim      : {sum(sims)/len(sims):.4f}  "
              f"(target < {FIELD_CONVENTION_THRESHOLD} field convention)")

    des = [r["defence_effectiveness"] for r in valid if "defence_effectiveness" in r]
    if des:
        print(f"  Mean defence effect.    : {sum(des)/len(des):.4f}  "
              f"(higher is better)")
    print("=" * 60)

    # ---- Save CSV --------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cloning_results.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"[CloningEval] Results saved to {csv_path}")

    return all_results
