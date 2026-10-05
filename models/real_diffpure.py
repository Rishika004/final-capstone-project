"""
models/real_diffpure.py
=======================
Real DiffPure purifier for mel spectrograms.

DiffPure (Nie et al., 2022) purifies adversarial examples by:
  1. Forward process : add Gaussian noise up to timestep t*
                       (partially destroys adversarial perturbation)
  2. Backward process: reverse-denoise using a pretrained score model
                       (reconstruct clean audio)

For audio, we implement this on mel spectrograms using our trained
MelUNetPurifier as the denoising backbone — but NOW running the full
DDPM forward+backward loop instead of a single pass.

Why this is different from our surrogate purifier
-------------------------------------------------
The surrogate purifier (models/purifier.py) does a SINGLE forward pass:
    x_clean_approx = purifier(x_noisy)

Real DiffPure does t* steps of noise addition then t* steps of denoising:
    x_t*   = sqrt(ab_t*) * x + sqrt(1-ab_t*) * eps    [forward]
    x_t*-1 = DDPM_step(x_t*, t*)                        [backward, step 1]
    ...
    x_0    = DDPM_step(x_1, 1)                          [backward, final]

The null space of the SINGLE-pass surrogate approximates the null space of
the full DDPM loop when t* is small (local linearity assumption).
This experiment TESTS whether that approximation holds in practice.

Usage
-----
    from models.real_diffpure import MelDiffPure, load_real_diffpure
    purifier = load_real_diffpure(device=device)
    mel_purified = purifier(mel_protected)    # same API as surrogate
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)

_this_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
CHECKPOINT_PATH = os.path.join(_project_root, "checkpoints", "purifier_trained.pt")


# ---------------------------------------------------------------------------
# Real DiffPure on mel spectrograms
# ---------------------------------------------------------------------------

class MelDiffPure(nn.Module):
    """
    DiffPure-style purifier operating on log-mel spectrograms.

    Uses the trained MelUNetPurifier as the denoising backbone but runs
    the FULL DDPM forward + backward loop (not a single pass).

    Parameters
    ----------
    denoiser : nn.Module
        Pretrained mel denoiser (MelUNetPurifier). Must accept [B, 80, T].
    n_steps : int
        Total DDPM timesteps (noise schedule length). Default 1000.
    t_diffuse : int
        Forward diffusion timestep t*. Controls how much noise is added.
        Smaller = less destructive, more faithful to original.
        Typical values: 50 (mild), 100 (moderate), 200 (aggressive).
    """

    def __init__(
        self,
        denoiser: nn.Module,
        n_steps: int = 1000,
        t_diffuse: int = 100,
        sampler: str = "ddpm",
    ):
        super().__init__()
        self.denoiser  = denoiser
        self.n_steps   = n_steps
        self.t_diffuse = t_diffuse
        # "ddpm": ancestral sampling (posterior mean + injected noise), as in
        #         the original DiffPure SDE formulation;
        # "ddim": deterministic reverse (eta=0). With an imperfect x0-predictor
        #         the injected DDPM noise accumulates (loop gain < 0 at small
        #         t, measured 2026-09-11); DDIM is the stronger purifier.
        assert sampler in ("ddpm", "ddim")
        self.sampler = sampler

        # ---------- DDPM linear beta schedule (Ho et al., 2020) ----------
        betas     = torch.linspace(1e-4, 0.02, n_steps)   # β_1 … β_T
        alphas    = 1.0 - betas                             # α_t = 1 - β_t
        alpha_bar = torch.cumprod(alphas, dim=0)            # ᾱ_t = ∏ α_s

        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        logger.info(
            "MelDiffPure | n_steps=%d | t_diffuse=%d | "
            "noise_level_at_t*=%.3f (sqrt(1-alpha_bar_t*))",
            n_steps, t_diffuse,
            float((1.0 - alpha_bar[t_diffuse - 1]).sqrt()),
        )

    # ------------------------------------------------------------------
    def forward(self, x_mel: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Apply DiffPure to a log-mel spectrogram.

        Parameters
        ----------
        x_mel : torch.Tensor, shape [B, n_mels, T]
        seed  : int, optional
            Fix the RNG seed so that all DiffPure calls within one experiment
            use identical forward/backward noise.  This makes the PSR metric
            interpretable: differences in output come only from differences in
            input, not from different noise draws.

        Returns
        -------
        x_purified : torch.Tensor, shape [B, n_mels, T]
        """
        if seed is not None:
            torch.manual_seed(seed)

        t   = self.t_diffuse
        # Schedule indexing: betas[k] is β_{k+1} (paper index k+1), so
        # ᾱ_t = alpha_bar[t-1]. The previous code used alpha_bar[t] and
        # betas[i] with ab_prev=1 at i=1, which made the final posterior
        # coefficient β_2/(1-ᾱ_2) ≈ 0.55 instead of 1 and SCALED THE OUTPUT
        # LOG-MEL BY ~0.55 (verified 2026-09-11 with an oracle denoiser:
        # 1.75 nats error at any t). Fixed to the standard DDPM indexing.
        abt = self.alpha_bar[t - 1]      # scalar ᾱ_{t*}

        # ---- Step 1: Forward — add noise up to timestep t* -------------
        eps   = torch.randn_like(x_mel)
        x_t   = abt.sqrt() * x_mel + (1.0 - abt).sqrt() * eps
        # x_t is now a noisy version that partially erases the perturbation

        # ---- Step 2: Backward — DDPM reverse from t* down to 0 --------
        x = x_t
        self.denoiser.eval()

        # NOTE: no torch.no_grad() here — keeping the loop differentiable
        # allows torch.autograd.functional.jvp to propagate tangents through
        # the full 100-step DDPM process when estimating the real-DiffPure
        # Jacobian for null-space computation.  Callers that want pure
        # inference should wrap the entire forward() call in torch.no_grad().
        for i in reversed(range(1, t + 1)):
            ab_i    = self.alpha_bar[i - 1]                      # ᾱ_i
            ab_prev = (self.alpha_bar[i - 2] if i > 1              # ᾱ_{i-1}, ᾱ_0 = 1
                       else torch.ones((), device=x.device))
            beta_i  = self.betas[i - 1]                          # β_i

            # ---- Predict x_0 from current x_i ----
            # Our denoiser was trained to output x_0 directly (not eps).
            x0_pred = self.denoiser(x)
            x0_pred = x0_pred.clamp(-10.0, 10.0)   # numerical safety

            # ---- DDPM posterior mean (Eq. 7 in Ho et al.) ----------
            # mu_q(x_t, x_0) = coef1 * x_0 + coef2 * x_t
            if self.sampler == "ddim":
                # eps implied by the x0 prediction, then deterministic step
                eps_pred = (x - ab_i.sqrt() * x0_pred) / (1.0 - ab_i).sqrt()
                x = ab_prev.sqrt() * x0_pred + (1.0 - ab_prev).sqrt() * eps_pred
                continue

            coef1 = (ab_prev.sqrt() * beta_i) / (1.0 - ab_i)
            coef2 = ((1.0 - ab_prev) * (1.0 - beta_i).sqrt()) / (1.0 - ab_i)
            mean  = coef1 * x0_pred + coef2 * x

            # ---- Add posterior noise (skip at final step i=1) ------
            if i > 1:
                var   = beta_i * (1.0 - ab_prev) / (1.0 - ab_i)
                sigma = var.clamp(min=1e-20).sqrt()
                x     = mean + sigma * torch.randn_like(x)
            else:
                x = mean   # deterministic at final step

        return x   # x_0: purified mel spectrogram


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_real_diffpure(
    checkpoint_path: str = CHECKPOINT_PATH,
    t_diffuse: int = 100,
    n_steps: int = 1000,
    device: Optional[torch.device] = None,
    sampler: str = "ddpm",
) -> MelDiffPure:
    """
    Load the trained MelUNetPurifier and wrap it as a real DiffPure purifier.

    Parameters
    ----------
    checkpoint_path : str
        Path to purifier_trained.pt (same checkpoint as surrogate).
    t_diffuse : int
        Forward diffusion steps. 100 ≈ mild DiffPure. 200 ≈ aggressive.
    device : torch.device

    Returns
    -------
    MelDiffPure (nn.Module, eval mode)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.purifier import MelUNetPurifier, MelUNet2D

    # Load denoiser backbone (same weights as surrogate purifier)
    residual = False
    ckpt = None
    arch, base = "unet1d", 64
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            residual = bool(ckpt.get("residual", False))
            arch = ckpt.get("arch", "unet1d")
            base = int(ckpt.get("base_channels", 64))
    if arch == "unet2d":
        denoiser = MelUNet2D(n_mels=80, base_channels=base, residual=residual).to(device)
    else:
        denoiser = MelUNetPurifier(n_mels=80, base_channels=base, residual=residual).to(device)

    if ckpt is not None:
        state = ckpt.get("model_state_dict", ckpt)
        denoiser.load_state_dict(state)
        print(f"[RealDiffPure] Loaded denoiser from: {checkpoint_path}")
    else:
        print(f"[RealDiffPure] WARNING: checkpoint not found at {checkpoint_path}")
        print("  Using randomly initialised denoiser — train first with --mode train")

    denoiser.eval()
    model = MelDiffPure(denoiser=denoiser, n_steps=n_steps, t_diffuse=t_diffuse, sampler=sampler)
    model = model.to(device)
    model.eval()

    noise_level = float((1.0 - model.alpha_bar[t_diffuse - 1]).sqrt())
    print(
        f"[RealDiffPure] Ready | t_diffuse={t_diffuse} | "
        f"forward_noise_level={noise_level:.3f} | device={device}"
    )
    return model
