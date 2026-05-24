"""
core/jacobian.py
================
Jacobian estimation for the surrogate purifier in mel-spectrogram space.

Why randomized estimation?
--------------------------
Full Jacobian computation (torch.autograd.functional.jacobian) requires one
backward pass per OUTPUT dimension.  For an 80×T mel spectrogram the output
has 80·T elements — for T=128 that is 10 240 backward passes, which is
prohibitively slow.

Randomized (Johnson-Lindenstrauss) estimator
--------------------------------------------
For a function f: R^n → R^m we approximate J = ∂f/∂x ∈ R^{m×n} via K
random projections:

    For k = 1 … K:
        1. Sample vₖ ~ Normal(0, I_n) (a random input direction)
        2. Compute Jvₖ via a single JVP (Jacobian-vector product) using
           torch.autograd.functional.jvp, which costs one forward + one
           tangent propagation.
        3. Stack {Jvₖ} column-by-column into Z ∈ R^{m×K}
        4. Solve for Ĵ ≈ Z · V^{-T}  where V = [v₁|…|vₖ] ∈ R^{n×K}

This gives a rank-K approximation sufficient for null-space extraction
when K ≪ effective_rank(J).

Memory note: for a 80×128 mel spectrogram (n=10240) with K=50, the
intermediate tensors are at most 50 × 10240 × 4 bytes ≈ 2 MB — fine for CPU.
"""

import logging
import time
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core estimation routine
# ---------------------------------------------------------------------------

def estimate_jacobian_randomized(
    model: nn.Module,
    x: torch.Tensor,
    K: int = 50,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build a rank-K approximation of the Jacobian of `model` at input `x`
    using randomized JVP estimation.

    Parameters
    ----------
    model : nn.Module
        Differentiable surrogate purifier.  Must accept tensors of the same
        shape as `x` and return tensors with a fixed output shape.
    x : torch.Tensor
        Input mel spectrogram, shape [batch, mel_bins, time] or [mel_bins, time].
        Gradients will be computed w.r.t. this tensor.
    K : int
        Number of random probe directions.  More probes → better approximation
        but linearly more compute.  K=50 is a good default for rank ≈ 10–30.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Z : torch.Tensor, shape [input_dim, K]
        JVP vectors: columns are the directional derivatives J·v_k.
        (Note: on CPU, moved from device for SVD efficiency.)
    V : torch.Tensor, shape [input_dim, K]
        Probe vectors: columns are the random probes v_k.

    The implicit Jacobian approximation is  J ≈ (1/K) * Z @ V^T.
    We return the factored form to avoid materializing the N×N matrix,
    which would be ~900 MB for a typical 3-second mel spectrogram.
    """
    device = x.device
    input_dim = x.numel()
    logger.info(
        "Randomized Jacobian estimation | K=%d | input_shape=%s | device=%s",
        K, list(x.shape), device,
    )

    model.eval()

    # Determine output dimension with a single noiseless forward pass
    with torch.no_grad():
        y0 = model(x)
    output_dim = y0.numel()
    logger.info("output_dim=%d | input_dim=%d", output_dim, input_dim)

    t0 = time.perf_counter()

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Accumulate factored form: Z=[Jv_1,...,Jv_K], V=[v_1,...,v_K]
    # Both have shape [input_dim, K] (we keep both on CPU to save VRAM)
    Z = torch.zeros(input_dim, K, dtype=torch.float32)  # JVP columns
    V = torch.zeros(input_dim, K, dtype=torch.float32)  # probe columns

    for k in range(K):
        v = torch.randn(input_dim, generator=rng, device=device)
        v_shaped = v.reshape(x.shape)

        _, jvp_val = torch.autograd.functional.jvp(
            func=lambda inp: model(inp),
            inputs=(x,),
            v=(v_shaped,),
            strict=False,
            create_graph=False,
        )

        jvp_flat = jvp_val.reshape(-1).detach().cpu().float()
        Z[:, k] = jvp_flat
        V[:, k] = v.detach().cpu().float()

    elapsed = time.perf_counter() - t0
    logger.info(
        "Jacobian estimation complete in %.2fs | Z=%s V=%s",
        elapsed, list(Z.shape), list(V.shape),
    )
    print(
        f"[Jacobian] K={K} probes | Z shape={list(Z.shape)} | "
        f"time={elapsed:.2f}s"
    )

    return Z, V


def compute_jacobian(
    model: nn.Module,
    input_tensor: torch.Tensor,
    method: str = "randomized",
    K: int = 50,
    seed: int = 42,
) -> torch.Tensor:
    """
    Unified entry point for Jacobian computation.

    Parameters
    ----------
    model : nn.Module
        Differentiable model (purifier surrogate).
    input_tensor : torch.Tensor
        Input mel spectrogram.
    method : str
        "randomized" (default, scalable) or "exact" (slow, for testing only).
    K : int
        Number of probes for the randomized method.
    seed : int
        RNG seed.

    Returns
    -------
    J : torch.Tensor, shape [output_dim, input_dim]
    """
    if method == "randomized":
        return estimate_jacobian_randomized(model, input_tensor, K=K, seed=seed)
    elif method == "exact":
        return _compute_jacobian_exact(model, input_tensor)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'randomized' or 'exact'.")


def _compute_jacobian_exact(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Exact Jacobian via torch.autograd.functional.jacobian.

    WARNING: O(output_dim) backward passes.  Only use for small inputs
    (e.g. 80×32 mel patches during unit tests).
    """
    logger.warning(
        "Computing EXACT Jacobian — this is O(output_dim) backward passes "
        "and will be slow for large inputs."
    )
    model.eval()

    J_full = torch.autograd.functional.jacobian(
        func=lambda inp: model(inp),
        inputs=x,
        strict=False,
        create_graph=False,
        vectorize=True,  # uses vmap under the hood for speed
    )

    # J_full shape: (*output_shape, *input_shape) — flatten to 2-D
    with torch.no_grad():
        out_numel = model(x).numel()
    input_dim = x.numel()

    J_2d = J_full.reshape(out_numel, input_dim)
    logger.info("Exact Jacobian shape: %s", list(J_2d.shape))
    return J_2d.float()
