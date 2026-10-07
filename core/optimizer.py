"""
core/optimizer.py
=================
Iterative null-space perturbation optimization.

Unlike the single-shot approach in perturbation.py, this module runs a
projected gradient descent loop that jointly optimises two objectives:

  1. Cloning loss   — maximise cosine distance between the speaker embedding
                      of the protected audio and the original embedding.
                      (Makes voice cloning fail.)

  2. Perceptual loss — minimize L2 distance in mel space between protected
                       and original audio.
                       (Keeps the audio sounding natural to humans.)

After every gradient step the perturbation is:
  a) Re-projected onto the null space of the purifier's Jacobian.
     (Makes the perturbation survive DiffPure-style denoising.)
  b) Clamped to the L∞ budget ε.

Algorithm (Projected Gradient Descent in the Null Space)
---------------------------------------------------------
    δ₀ = 0
    for t = 1 … T:
        g  = ∇_δ [λ_clone · L_clone(δ) + λ_percept · L_percept(δ)]
        δ  = δ  -  lr · g                  # gradient step
        δ  = P_null(δ)                     # project onto null(J)
        δ  = clip(δ, -ε, +ε)              # L∞ constraint
    return x_wav + wav_from_mel(δ)

The Jacobian J is computed once at the start (it changes slowly) and reused
across all optimisation steps, which keeps compute tractable.
"""

import logging
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
try:
    import torchaudio.transforms as T
except Exception:   # Smart App Control blocks libtorchaudio.pyd (2026-09-15)
    from utils import tacompat as T

from core.jacobian import estimate_jacobian_randomized
from core.nullspace import compute_null_space_factored, project_onto_null_space


def _build_mel_to_wav(
    n_fft: int,
    hop_length: int,
    n_mels: int,
    sample_rate: int,
    x_wav: torch.Tensor,
    device: torch.device,
):
    """
    Precompute everything needed for a differentiable mel → wav reconstruction.

    Strategy: use the *original* waveform's STFT phase (fixed) so that
    InverseMelScale(perturbed_mel) × phase_orig fed through iSTFT gives a
    waveform whose gradient w.r.t. delta flows cleanly through:

        delta → mel_lin (exp) → spec_mag (InverseMelScale) → spec_complex
              → wav (iSTFT) → WavLM embedding → cosine loss

    Phase is detached — it is a constant w.r.t. delta — so no extra memory
    cost.  InverseMelScale uses a least-squares solve (differentiable via
    torch.linalg.lstsq).

    Returns (inv_mel_scale, phase_orig, window, orig_wav_len, T_frames).
    """
    window = torch.hann_window(n_fft, device=device)
    with torch.no_grad():
        stft_orig = torch.stft(
            x_wav.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )  # [n_fft//2+1, T_frames]
        phase_orig = (stft_orig / (stft_orig.abs() + 1e-8)).detach()

    inv_mel_scale = T.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        driver="gels",   # gelsd not supported on CUDA; gels works on both CPU and GPU
    ).to(device)

    return inv_mel_scale, phase_orig, window, x_wav.shape[-1], stft_orig.shape[-1]


def _mel_to_wav_differentiable(
    mel_perturbed: torch.Tensor,
    inv_mel_scale: T.InverseMelScale,
    phase_orig: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    T_frames: int,
    orig_wav_len: int,
) -> torch.Tensor:
    """
    Differentiable mel → approx waveform reconstruction.

    mel_perturbed : [1, n_mels, T_mel]  (log-mel)
    Returns       : [1, T_wav]
    """
    mel_lin = torch.exp(mel_perturbed).clamp(min=1e-9)   # undo log, [1, n_mels, T_mel]
    spec_mag = inv_mel_scale(mel_lin.squeeze(0))          # [n_fft//2+1, T_spec]

    # Align T_spec → T_frames
    T_spec = spec_mag.shape[-1]
    if T_spec > T_frames:
        spec_mag = spec_mag[..., :T_frames]
    elif T_spec < T_frames:
        spec_mag = nn.functional.pad(spec_mag, (0, T_frames - T_spec))

    spec_complex = spec_mag * phase_orig                  # apply original phase
    wav = torch.istft(
        spec_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=orig_wav_len,
    )                                                     # [T_wav]
    return wav.unsqueeze(0)                               # [1, T_wav]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio ↔ Mel helpers (self-contained so this module has no circular imports)
# ---------------------------------------------------------------------------

_MEL_TRANSFORM_CACHE: dict = {}


def _get_mel_transform(
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    device: torch.device = torch.device("cpu"),
) -> T.MelSpectrogram:
    key = (sample_rate, n_mels, n_fft, hop_length, str(device))
    if key not in _MEL_TRANSFORM_CACHE:
        _MEL_TRANSFORM_CACHE[key] = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ).to(device)
    return _MEL_TRANSFORM_CACHE[key]


def _wav_to_logmel(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """Convert waveform [1, T] → log-mel spectrogram [1, n_mels, frames]."""
    mel_transform = _get_mel_transform(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        device=wav.device,
    )
    mel = mel_transform(wav)
    log_mel = torch.log(mel + 1e-9)
    return log_mel


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _cloning_loss(
    wav_perturbed: torch.Tensor,
    emb_original: torch.Tensor,
    speaker_encoder: nn.Module,
) -> torch.Tensor:
    """
    Cloning loss: cosine similarity between embeddings (to be minimised).

    We want the protected audio to sound like a *different* speaker, so we
    minimise cosine similarity (= maximise cosine distance).

    L_clone = cosine_similarity(embed(x_protected), embed(x_original))
    Gradient descent on L_clone pushes embeddings apart → cloning fails.
    """
    emb_protected = speaker_encoder(wav_perturbed)
    # Normalise to unit sphere
    emb_p = nn.functional.normalize(emb_protected, dim=-1)
    emb_o = nn.functional.normalize(emb_original, dim=-1)
    cos_sim = (emb_p * emb_o).sum(dim=-1)  # [batch] or scalar
    return cos_sim.mean()  # minimise = push embeddings apart = cloning fails


def _energy_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    Energy maximisation loss: negative RMS of the mel perturbation.

    We WANT delta to be as large as possible (up to the L∞ budget epsilon).
    Minimising the negative RMS pushes delta toward the epsilon boundary.
    Perceptual quality is bounded by the L∞ clamp — no separate MSE term needed.

    L_energy = -RMS(delta)   [minimise → delta grows]
    """
    return -(delta ** 2).mean().sqrt()


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------

def optimize_perturbation(
    x_mel: torch.Tensor,
    purifier: nn.Module,
    speaker_encoder: nn.Module,
    x_wav: torch.Tensor,
    n_steps: int = 100,
    lr: float = 0.01,
    lambda_clone: float = 1.0,
    lambda_percept: float = 0.5,
    epsilon: float = 0.01,
    K: int = 50,
    svd_threshold: float = 1e-3,
    seed: int = 42,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    log_interval: int = 10,
) -> Tuple[torch.Tensor, dict]:
    """
    Run iterative null-space projected gradient descent.

    Parameters
    ----------
    x_mel : torch.Tensor
        Original log-mel spectrogram, shape [1, n_mels, time].
    purifier : nn.Module
        Differentiable surrogate purifier.
    speaker_encoder : nn.Module
        Pretrained speaker encoder (e.g. WavLM).  Must accept raw waveforms.
    x_wav : torch.Tensor
        Original waveform, shape [1, T].  Used for speaker embedding.
    n_steps : int
        Number of optimisation steps.
    lr : float
        Learning rate for the gradient update.
    lambda_clone : float
        Weight for the cloning (speaker distance) loss.
    lambda_percept : float
        Weight for the perceptual (mel distance) loss.
    epsilon : float
        L∞ perturbation budget.
    K : int
        Jacobian probe count.
    svd_threshold : float
        SVD singular value cutoff for null space.
    seed : int
        RNG seed.
    sample_rate, n_mels, n_fft, hop_length : int
        Audio processing parameters (must match purifier's training config).
    log_interval : int
        Print loss every this many steps.

    Returns
    -------
    x_protected_wav : torch.Tensor
        Protected waveform, shape [1, T].
    history : dict
        {'total_loss': [...], 'clone_loss': [...], 'energy_loss': [...]}
    """
    device = x_mel.device
    logger.info(
        "optimize_perturbation | n_steps=%d | lr=%.4f | epsilon=%.4f | "
        "lambda_clone=%.2f | lambda_percept=%.2f",
        n_steps, lr, epsilon, lambda_clone, lambda_percept,
    )

    purifier.eval()
    speaker_encoder.eval()

    # ------------------------------------------------------------------
    # Step 1: Compute Jacobian once (reused across all gradient steps)
    # ------------------------------------------------------------------
    print("[Optimizer] Computing Jacobian (this may take a moment) ...")
    Z, V = estimate_jacobian_randomized(purifier, x_mel, K=K, seed=seed)
    null_basis, _ = compute_null_space_factored(Z, V, threshold=svd_threshold)

    if null_basis.shape[0] == 0:
        logger.error("Null space is empty — cannot run optimisation.")
        return x_wav.clone(), {}

    # Keep null_basis on CPU — for long audio it can be [48270 × 48320] ≈ 9 GB,
    # which exceeds GPU VRAM. project_onto_null_space() handles the CPU↔GPU
    # transfer internally (projects on CPU, returns result on original device).
    print(f"[Optimizer] Null space dim = {null_basis.shape[0]} | projection on CPU")

    # ------------------------------------------------------------------
    # Step 1b: Build differentiable mel → wav chain (fixed-phase iSTFT)
    #
    # WHY: the speaker encoder (WavLM) needs a waveform, not a mel spectrogram.
    # We reconstruct a waveform from the perturbed mel by:
    #   exp(mel + delta) → InverseMelScale → magnitude spectrogram
    #   → multiply by original STFT phase (fixed) → iSTFT → waveform
    #
    # This gives a proper gradient path:
    #   loss ← WavLM ← wav_perturbed ← iSTFT ← InverseMelScale ← delta
    #
    # The old proxy (fixed_noise_wav * mel_rms) only provided a scalar
    # gradient that told delta to grow/shrink but gave no directional
    # information about WHICH null-space direction confuses WavLM.
    # ------------------------------------------------------------------
    print("[Optimizer] Precomputing STFT phase for differentiable mel->wav ...")
    inv_mel_scale, phase_orig, stft_window, orig_wav_len, T_frames = _build_mel_to_wav(
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        sample_rate=sample_rate,
        x_wav=x_wav,
        device=device,
    )

    # ------------------------------------------------------------------
    # Step 2: Pre-compute original speaker embedding (no-grad, fixed target)
    # ------------------------------------------------------------------
    with torch.no_grad():
        emb_original = speaker_encoder(x_wav)
        emb_original = nn.functional.normalize(emb_original, dim=-1).detach()

    # ------------------------------------------------------------------
    # Step 3: Initialise perturbation δ in mel space as a leaf variable
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    delta_init = torch.randn_like(x_mel) * (epsilon * 0.01)
    delta = delta_init.detach().requires_grad_(True)
    optimizer_obj = torch.optim.Adam([delta], lr=lr)

    history = {"total_loss": [], "clone_loss": [], "energy_loss": []}

    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        optimizer_obj.zero_grad()

        # --- Differentiable mel → wav reconstruction ---
        # Gradient path: delta → mel_perturbed → mel_lin (exp) → spec_mag
        #   (InverseMelScale, lstsq) → spec_complex (* fixed phase)
        #   → wav_perturbed (iSTFT) → WavLM → cosine loss
        mel_perturbed = x_mel + delta
        wav_perturbed = _mel_to_wav_differentiable(
            mel_perturbed=mel_perturbed,
            inv_mel_scale=inv_mel_scale,
            phase_orig=phase_orig,
            window=stft_window,
            n_fft=n_fft,
            hop_length=hop_length,
            T_frames=T_frames,
            orig_wav_len=orig_wav_len,
        )

        # --- Losses ---
        l_clone = _cloning_loss(wav_perturbed, emb_original, speaker_encoder)
        l_energy = _energy_loss(delta)

        loss = lambda_clone * l_clone + lambda_percept * l_energy
        loss.backward()

        optimizer_obj.step()

        # --- Null-space projection (in-place) ---
        with torch.no_grad():
            delta_proj = project_onto_null_space(delta.data, null_basis, kind="range")
            delta_proj = delta_proj.clamp(-epsilon, epsilon)
            delta.data.copy_(delta_proj)

        # --- Logging ---
        if step % log_interval == 0 or step == 1:
            history["total_loss"].append(loss.item())
            history["clone_loss"].append(l_clone.item())
            history["energy_loss"].append(l_energy.item())

            elapsed = time.perf_counter() - t0
            print(
                f"[Optimizer] step {step:4d}/{n_steps} | "
                f"loss={loss.item():.4f} | "
                f"L_clone={l_clone.item():.4f} | "
                f"L_energy={l_energy.item():.5f} | "
                f"||d||_inf={delta.data.abs().max().item():.5f} | "
                f"t={elapsed:.1f}s"
            )

    # ------------------------------------------------------------------
    # Step 4: Build final protected waveform via Griffin-Lim inversion
    # ------------------------------------------------------------------
    with torch.no_grad():
        final_delta = delta.data.clamp(-epsilon, epsilon)
        mel_protected = x_mel + final_delta   # [1, n_mels, T]

    # Use Griffin-Lim to invert the protected mel back to a waveform.
    # This correctly maps the null-space mel perturbation into the audio output,
    # unlike the previous approach (scaled random noise) which was decoupled from delta.
    from utils.vocoder import invert_mel
    with torch.no_grad():
        x_protected_wav = invert_mel(mel_protected.cpu(), method="griffin_lim")
        x_protected_wav = x_protected_wav.to(device)
        # Griffin-Lim may produce slightly different length — align to original
        orig_len = x_wav.shape[-1]
        prot_len = x_protected_wav.shape[-1]
        if prot_len > orig_len:
            x_protected_wav = x_protected_wav[..., :orig_len]
        elif prot_len < orig_len:
            x_protected_wav = nn.functional.pad(x_protected_wav, (0, orig_len - prot_len))

    logger.info(
        "Optimisation complete | final_loss=%.4f | ||d||_inf=%.5f",
        history["total_loss"][-1] if history["total_loss"] else float("nan"),
        final_delta.abs().max().item(),
    )
    print("[Optimizer] Done.")

    return x_protected_wav, history
