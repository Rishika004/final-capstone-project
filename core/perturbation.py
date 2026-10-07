"""
core/perturbation.py
====================
Null-space perturbation generation — two modes:

  MODE 1: random  (default, fast)
  ─────────────────────────────────────────────────────
  Generate a random delta, project onto null(J), clamp to epsilon.
  Fast but doesn't optimise for speaker confusion.

  MODE 2: pgd  (stronger protection)
  ─────────────────────────────────────────────────────
  PGD (Projected Gradient Descent) adversarial attack on the speaker
  encoder in WAVEFORM space, then convert delta to mel space and project
  onto null(J).

  Why waveform space for PGD?
    - Speaker encoders (WavLM) operate on raw waveforms.
    - The mel→wav path (Griffin-Lim) is not differentiable.
    - Gradients flow cleanly: WavLM ← wav_perturbed ← wav_delta.
    - After PGD we convert delta_wav → delta_mel via the mel transform,
      then project onto null(J) in mel space as usual.

  Combined pipeline (PGD mode):
    1. Estimate Jacobian of purifier → null space basis
    2. PGD in waveform space to fool speaker encoder (50 steps)
    3. Convert adversarial waveform delta → mel-space delta
    4. Project mel delta onto null(J)  ← survives DiffPure
    5. Clamp to epsilon                ← imperceptible
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
try:
    import torchaudio.transforms as T
except Exception:   # Smart App Control blocks libtorchaudio.pyd (2026-09-15)
    from utils import tacompat as T

from core.jacobian import estimate_jacobian_randomized
from core.nullspace import (
    compute_null_space_factored,
    project_onto_null_space,
    verify_null_space_factored,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode 1: Random base perturbation
# ---------------------------------------------------------------------------

def generate_base_perturbation(
    x_mel: torch.Tensor,
    epsilon: float = 0.01,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate a random L-inf bounded perturbation in mel-spectrogram space.

    Parameters
    ----------
    x_mel   : reference tensor (shape / device / dtype only)
    epsilon : L-inf budget
    seed    : optional RNG seed

    Returns
    -------
    delta : torch.Tensor, same shape as x_mel, values in [-epsilon, +epsilon]
    """
    if seed is not None:
        generator = torch.Generator(device=x_mel.device)
        generator.manual_seed(seed)
        delta = (torch.rand(x_mel.shape, generator=generator,
                            device=x_mel.device) * 2 - 1) * epsilon
    else:
        delta = (torch.rand_like(x_mel) * 2 - 1) * epsilon

    delta = delta.clamp(-epsilon, epsilon)
    logger.debug("Base perturbation | shape=%s | L_inf=%.5f",
                 list(delta.shape), delta.abs().max().item())
    return delta


# ---------------------------------------------------------------------------
# Mode 2: PGD adversarial perturbation
# ---------------------------------------------------------------------------

def generate_pgd_perturbation(
    x_mel: torch.Tensor,
    x_wav: torch.Tensor,
    speaker_encoder: nn.Module,
    epsilon: float = 0.01,
    n_steps: int = 50,
    lr: float = 2e-3,
    seed: int = 42,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    null_basis: Optional[torch.Tensor] = None,
    lambda_null: float = 5.0,
    perceptual: bool = True,
    masking_margin_db: float = 25.0,
    bandlimit_hz: Optional[float] = 6000.0,
) -> torch.Tensor:
    """
    Two-stage PGD: waveform attack -> mel-space projection.

    When ``perceptual=True`` (default) Stage 1 uses Adam and, after every
    step, projects the waveform delta under the frequency-masking threshold
    of the original speech (core/psychoacoustic.py). The delta then hides
    beneath the voice's own spectral energy instead of being flat +-0.05
    broadband hiss, so the protected audio sounds like the original.
    ``masking_margin_db`` raises the cap (+6 dB = 2x amplitude = stronger but
    slightly more audible). ``perceptual=False`` restores the old flat
    signed-gradient behaviour. ``bandlimit_hz`` additionally removes delta
    energy above the cutoff (robustness to resampling / low-pass).

    Stage 1 — Joint waveform PGD against WavLM + null-space constraint:
        Run signed-gradient PGD in waveform space with a combined loss:
          L = cos_sim(WavLM(wav+δ), WavLM(wav))          ← minimise speaker similarity
            + λ_null * ||P_range(mel(wav+δ) - mel(wav))||  ← keep mel delta in null space

        The null-space constraint (if null_basis is provided) forces the mel
        delta to stay in DiffPure's null space *during* optimisation, so the
        waveform bypass and the null-space-projected mel become consistent.
        λ_null=5.0 trades ~10% less speaker confusion for reliable null-space
        alignment. Without this, PGD in waveform space gives a mel delta that
        lies partly in DiffPure's range space and gets stripped.

    Stage 2 — Mel-space conversion with magnitude-preserving normalisation:
        Convert the waveform delta to mel space.
        Normalise so the strongest mel bin hits epsilon — preserving
        the RELATIVE magnitudes across all bins (not just the sign).

    Parameters
    ----------
    x_mel          : original log-mel spectrogram [1, n_mels, T_mel]
    x_wav          : original waveform [1, T_wav]
    speaker_encoder: pretrained WavLM (operates on raw wav)
    epsilon        : L-inf budget in MEL space
    n_steps        : PGD iterations (Stage 1)
    lr             : signed-gradient step size
    seed           : RNG seed
    null_basis     : null space basis vectors [null_dim, mel_flat] (optional).
                     When provided, adds a range-space penalty to the PGD loss
                     so the mel delta of wav_delta lies in DiffPure's null space.
    lambda_null    : weight for the null-space constraint (default 5.0)

    Returns
    -------
    delta_mel : torch.Tensor, same shape as x_mel, in [-epsilon, +epsilon]
    best_wav_delta : torch.Tensor, same shape as x_wav — the raw Stage-1
        adversarial waveform delta (already fooled WavLM to cos_sim~-0.47).
        Callers that want to bypass Griffin-Lim can add this directly to the
        original waveform: ``(x_wav + best_wav_delta).clamp(-1, 1)``
    """
    device = x_wav.device
    speaker_encoder.eval()

    # Waveform L-inf budget.
    # Must be large enough that log_mel(wav+delta) - log_mel(wav) is well
    # above floating-point noise.  0.05 = 5% of signal range is still
    # imperceptible at audio sample level and gives a clean mel gradient.
    epsilon_wav = 0.05

    # Mel transform — used both for null-space constraint (Stage 1) and
    # mel-delta conversion (Stage 2).  Build once, reuse.
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
    ).to(device)

    # Reference log-mel for null-space constraint computation
    with torch.no_grad():
        log_mel_ref = torch.log(mel_transform(x_wav) + 1e-9)   # [1, n_mels, T_mel]
        T_mel_target = x_mel.shape[-1]
        if log_mel_ref.shape[-1] > T_mel_target:
            log_mel_ref = log_mel_ref[..., :T_mel_target]
        elif log_mel_ref.shape[-1] < T_mel_target:
            log_mel_ref = nn.functional.pad(log_mel_ref, (0, T_mel_target - log_mel_ref.shape[-1]))
        mel_flat_size = log_mel_ref.numel()

    # Pre-process null basis for the optional joint-optimisation path.
    # null_basis is the RANGE basis [K, mel_flat] from compute_null_space_factored.
    # Constraint: penalise the RANGE component of the mel delta so the wav delta's
    # mel stays in DiffPure's null space.  Disabled when null_basis=None.
    null_b = None
    if null_basis is not None:
        null_b = null_basis.to(device)
        if null_b.shape[-1] != mel_flat_size:
            logger.warning(
                "null_basis last-dim %d != mel_flat_size %d — disabling null-space constraint.",
                null_b.shape[-1], mel_flat_size,
            )
            null_b = None
        else:
            logger.info("Joint PGD | null-space constraint enabled | "
                        "lambda_null=%.2f | K=%d", lambda_null, null_b.shape[0])
            print(f"  [PGD] Joint optimisation: WavLM confusion + null-space constraint "
                  f"(lambda={lambda_null:.2f}, K={null_b.shape[0]})")

    # ── Stage 1: Signed-gradient PGD in waveform space ───────────────────
    with torch.no_grad():
        emb_orig = nn.functional.normalize(
            speaker_encoder(x_wav), dim=-1
        ).detach()

    torch.manual_seed(seed)
    # Initialise inside the epsilon ball
    wav_delta = (torch.rand_like(x_wav) * 2 - 1) * epsilon_wav * 0.1
    wav_delta = wav_delta.detach().requires_grad_(True)

    # Psychoacoustic masking setup (perceptual mode): per-bin magnitude cap
    # computed once from the ORIGINAL audio; the delta is re-projected under
    # it after every optimiser step.
    masker = None
    mask_cap = None
    adam_opt: Optional[torch.optim.Adam] = None
    if perceptual:
        from core.psychoacoustic import PsychoacousticMasker, project_delta_to_masking
        masker = PsychoacousticMasker(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            margin_db=masking_margin_db, device=device,
        )
        mask_cap = masker.magnitude_cap(x_wav)
        # Small RANDOM init, not zeros. At delta=0 the loss cos(f(x+d), f(x))
        # sits at its maximum (=1.0), so its gradient vanishes (~2e-8) and Adam
        # never leaves the origin. A tiny random kick lands off that stationary
        # point where the gradient is healthy (~1e-2). Kept far under the
        # masking cap so it does not affect final quality.
        wav_delta = ((torch.rand_like(x_wav) * 2 - 1) * epsilon_wav * 0.05)
        wav_delta = wav_delta.detach().requires_grad_(True)
        # lr=2e-3 matches the validated margin sweep (exp_margin_sweep.py).
        adam_opt = torch.optim.Adam([wav_delta], lr=2e-3)
        # Masking-projected Adam needs more iterations than signed-gradient PGD
        # to converge under the constraint; the validated sweet spot used ~200.
        if n_steps < 150:
            n_steps = 150

    # Signed-gradient (FGSM-style) PGD: each step takes the maximum
    # allowed step in the gradient direction within the L-inf ball.
    # More reliable than Adam for adversarial attacks on speaker encoders.
    step_size = epsilon_wav / max(n_steps // 2, 1)   # reach boundary in ~half the steps

    null_tag = " + null-space" if null_b is not None else ""
    mode_tag = ("Adam + masking projection" if perceptual
                else "signed-gradient")
    print(f"  [PGD Stage 1] {n_steps} {mode_tag} steps | "
          f"epsilon_wav={epsilon_wav:.3f}{null_tag}")
    best_cos   = float("inf")
    best_delta = wav_delta.detach().clone()

    for step in range(1, n_steps + 1):
        if adam_opt is not None:
            adam_opt.zero_grad()
        elif wav_delta.grad is not None:
            wav_delta.grad.zero_()

        wav_perturbed = x_wav.detach() + wav_delta
        emb_prot = nn.functional.normalize(
            speaker_encoder(wav_perturbed), dim=-1
        )
        cos_sim = (emb_prot * emb_orig).sum()   # minimise = push apart

        # ── Null-space constraint (joint PGD) ────────────────────────────
        # null_b is the RANGE BASIS [K, mel_flat] returned by
        # compute_null_space_factored — it spans the K directions DiffPure
        # CAN modify.  The null-space projection is:
        #   P_null(δ) = δ - null_b.T @ (null_b @ δ)
        # We penalise the RANGE component so the mel delta stays invisible
        # to DiffPure:
        #   loss_null = ||null_b.T @ (null_b @ δ_mel)||  (range component)
        if null_b is not None:
            mel_pert = torch.log(mel_transform(wav_perturbed) + 1e-9)
            # Align time frames
            if mel_pert.shape[-1] > T_mel_target:
                mel_pert = mel_pert[..., :T_mel_target]
            elif mel_pert.shape[-1] < T_mel_target:
                mel_pert = nn.functional.pad(mel_pert, (0, T_mel_target - mel_pert.shape[-1]))
            delta_flat  = (mel_pert - log_mel_ref).reshape(-1)        # [mel_flat]
            # range_coords: how much of delta is in DiffPure's active subspace
            range_coords = null_b @ delta_flat                         # [K]
            # range_proj: the component DiffPure WILL erase (minimise this)
            range_proj   = range_coords @ null_b                       # [mel_flat]
            loss_null    = range_proj.norm()
            total_loss   = cos_sim + lambda_null * loss_null
        else:
            total_loss = cos_sim
        # ─────────────────────────────────────────────────────────────────

        total_loss.backward()

        if adam_opt is not None:
            # cos_sim was computed with the CURRENT (already projected) delta,
            # so snapshot the best BEFORE stepping.
            if cos_sim.item() < best_cos:
                best_cos   = cos_sim.item()
                best_delta = wav_delta.detach().clone()
            adam_opt.step()
            with torch.no_grad():
                from core.psychoacoustic import project_delta_to_masking
                proj = project_delta_to_masking(
                    wav_delta.data, mask_cap, masker,
                    bandlimit_hz=bandlimit_hz, hard_linf=epsilon_wav,
                ).to(device)
                wav_delta.data.copy_(proj)
        else:
            with torch.no_grad():
                # Signed gradient step: move in the direction that minimises cos_sim
                wav_delta.data -= step_size * wav_delta.grad.sign()
                # Project back onto L-inf ball
                wav_delta.data.clamp_(-epsilon_wav, epsilon_wav)

            if cos_sim.item() < best_cos:
                best_cos   = cos_sim.item()
                best_delta = wav_delta.detach().clone()

        if step % 10 == 0 or step == 1:
            null_str = (f" | range_loss={loss_null.item():.6f}"
                        if null_b is not None else "")
            print(f"  [PGD] step {step:3d}/{n_steps} | "
                  f"cos_sim={cos_sim.item():.4f} | "
                  f"||dw||_inf={wav_delta.data.abs().max().item():.5f}{null_str}")

    print(f"  [PGD Stage 1] Done | best cos_sim={best_cos:.4f}")

    # ── Stage 2: Convert to mel space with magnitude-preserving scale ─────
    #
    # OLD (broken): sign(delta_mel) * epsilon
    #   - Every mel bin gets weight ±epsilon regardless of how much it changed
    #   - At epsilon_wav=0.003 sign() was noise anyway
    #
    # NEW (fixed): delta_mel / max(|delta_mel|) * epsilon
    #   - Bin that changed most gets weight ±epsilon
    #   - All other bins scaled proportionally
    #   - Preserves the actual adversarial fingerprint from Stage 1
    # ─────────────────────────────────────────────────────────────────────
    # Note: mel_transform was already created above (for null-space constraint)

    with torch.no_grad():
        mel_orig_lin = mel_transform(x_wav)
        mel_prot_lin = mel_transform((x_wav + best_delta).clamp(-1.0, 1.0))
        log_mel_orig = torch.log(mel_orig_lin + 1e-9)
        log_mel_prot = torch.log(mel_prot_lin + 1e-9)
        delta_mel    = log_mel_prot - log_mel_orig

        # Align time frames
        T_target = x_mel.shape[-1]
        T_got    = delta_mel.shape[-1]
        if T_got > T_target:
            delta_mel = delta_mel[..., :T_target]
        elif T_got < T_target:
            delta_mel = nn.functional.pad(delta_mel, (0, T_target - T_got))

        # Magnitude-preserving normalisation to epsilon budget
        max_abs = delta_mel.abs().max()
        if max_abs > 1e-6:
            delta_mel = (delta_mel / max_abs) * epsilon   # scale to epsilon, keep ratios
        else:
            # Fallback: if mel delta is still tiny, use sign (shouldn't happen at eps=0.05)
            logger.warning("Mel delta is near-zero (max=%.2e) — falling back to sign.", max_abs.item())
            delta_mel = delta_mel.sign() * epsilon

        delta_mel = delta_mel.clamp(-epsilon, epsilon)

    mel_inf = delta_mel.abs().max().item()
    mel_rms = delta_mel.pow(2).mean().sqrt().item()
    print(f"  [PGD Stage 2] Mel delta scaled to epsilon={epsilon:.4f} | "
          f"||dm||_inf={mel_inf:.5f} | RMS={mel_rms:.5f} | "
          f"adversarial direction preserved (magnitudes kept)")

    logger.info("PGD perturbation | best_cos_sim=%.4f | "
                "||delta_mel||_inf=%.5f | rms=%.5f",
                best_cos, mel_inf, mel_rms)
    return delta_mel.cpu(), best_delta.cpu()


# ---------------------------------------------------------------------------
# Main protection pipeline (random or pgd)
# ---------------------------------------------------------------------------

def apply_null_space_protection(
    x_mel: torch.Tensor,
    purifier: nn.Module,
    epsilon: float = 0.01,
    K: int = 50,
    svd_threshold: float = 1e-3,
    seed: int = 42,
    verify: bool = True,
    _precomputed_null_basis: Optional[torch.Tensor] = None,
    # PGD-specific parameters
    mode: str = "random",
    x_wav: Optional[torch.Tensor] = None,
    speaker_encoder: Optional[nn.Module] = None,
    pgd_steps: int = 50,
    pgd_lr: float = 5e-4,
    perceptual: bool = True,
    masking_margin_db: float = 25.0,
    bandlimit_hz: Optional[float] = 6000.0,
) -> "tuple[torch.Tensor, Optional[torch.Tensor]]":
    """
    Apply null-space protection to a mel spectrogram.

    Parameters
    ----------
    x_mel        : log-mel spectrogram [1, n_mels, time]
    purifier     : differentiable surrogate purifier
    epsilon      : L-inf budget in mel space
    K            : Jacobian probe count
    svd_threshold: singular value cutoff
    seed         : RNG seed
    verify       : print null-space verification residual
    _precomputed_null_basis : skip Jacobian if already computed
    mode         : "random" (fast) or "pgd" (stronger, needs x_wav + encoder)
    x_wav        : original waveform [1, T] — required for pgd mode
    speaker_encoder : WavLM or similar — required for pgd mode
    pgd_steps    : PGD iterations
    pgd_lr       : PGD Adam learning rate

    Returns
    -------
    x_protected : torch.Tensor — protected log-mel spectrogram (same shape as x_mel)
    best_wav_delta : Optional[torch.Tensor] — raw Stage-1 waveform delta (pgd mode
        only; None in random mode).  Add directly to the original waveform to get
        a protected audio signal that bypasses Griffin-Lim phase reconstruction.
    """
    logger.info(
        "apply_null_space_protection | mode=%s | epsilon=%.4f | K=%d",
        mode, epsilon, K,
    )
    purifier.eval()

    # ── Step 1-2: Null space (reuse pre-computed if available) ────────────
    if _precomputed_null_basis is not None:
        null_basis = _precomputed_null_basis
        print("[Protection] Using pre-computed null basis (skipping Jacobian step)")
    else:
        print("[Protection] Step 1/5 - Estimating Jacobian ...")
        Z, V = estimate_jacobian_randomized(purifier, x_mel, K=K, seed=seed)

        print("[Protection] Step 2/5 - Extracting null space via factored SVD ...")
        null_basis, _ = compute_null_space_factored(Z, V, threshold=svd_threshold)

        if verify and null_basis.shape[0] > 0:
            verify_null_space_factored(V, null_basis)

    if null_basis.shape[0] == 0:
        logger.error("Null space is empty — returning original mel.")
        return x_mel.clone(), None

    # ── Step 3: Generate base perturbation (random or PGD) ───────────────
    best_wav_delta: Optional[torch.Tensor] = None

    if mode == "pgd":
        if x_wav is None or speaker_encoder is None:
            logger.warning("PGD mode requires x_wav and speaker_encoder. "
                           "Falling back to random.")
            mode = "random"

    if mode == "pgd":
        print("[Protection] Step 3/5 - PGD adversarial perturbation ...")
        delta, best_wav_delta = generate_pgd_perturbation(
            x_mel=x_mel,
            x_wav=x_wav,
            speaker_encoder=speaker_encoder,
            epsilon=epsilon,
            n_steps=pgd_steps,
            lr=pgd_lr,
            seed=seed,
            perceptual=perceptual,
            masking_margin_db=masking_margin_db,
            bandlimit_hz=bandlimit_hz,
            # null_basis not passed here — null-space projection happens in Step 4.
            # Joint optimisation via null_basis degrades speaker confusion because
            # range_loss dominates cos_sim gradient.  Keep them separate.
        )
    else:
        print("[Protection] Step 3/5 - Generating base perturbation ...")
        delta = generate_base_perturbation(x_mel.cpu(), epsilon=epsilon, seed=seed)

    # ── Step 4: Project onto null space ──────────────────────────────────
    print("[Protection] Step 4/5 - Projecting onto null space ...")
    delta_proj = project_onto_null_space(delta, null_basis, kind="range")

    # ── Step 5: Clamp to epsilon budget ──────────────────────────────────
    print(f"[Protection] Step 5/5 - Clamping to L-inf budget e={epsilon} ...")
    delta_proj = delta_proj.clamp(-epsilon, epsilon)

    x_protected = x_mel + delta_proj.to(x_mel.device)

    logger.info("Protection done | ||delta||_inf=%.5f | budget=%.4f",
                delta_proj.abs().max().item(), epsilon)
    print(f"[Protection] Done. "
          f"||delta||_inf = {delta_proj.abs().max().item():.5f} (budget e={epsilon})")
    return x_protected, best_wav_delta
