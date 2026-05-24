"""
core/nullspace.py
=================
Null space extraction and projection via Singular Value Decomposition (SVD).

Two modes
---------
1. compute_null_space(J)          — classic: SVD of the full [output, input] matrix J.
                                    Used in unit tests and when J is small.

2. compute_null_space_factored(Z, V) — SCALABLE: SVD of V.T [K, input_dim].
                                    Used by the main pipeline for real audio.

Why the factored form is required for real audio
------------------------------------------------
A 3-second mel clip at 16 kHz / hop 256 has ~188 time frames.
  input_dim = 80 * 188 = 15 040
Full J matrix: 15040 x 15040 = 900 MB.
SVD of full J: O(N^3) ≈ hours on CPU.

Factored form: J ≈ (1/K) Z V^T  where Z, V are [input_dim, K] with K=50.
SVD of V.T [K, input_dim]: O(K^2 * N) ≈ 37 M ops  → under a second.

Mathematical validity
---------------------
null(J) ≈ null(V^T) when Z has full column rank (true for random Gaussian probes).
Proof:  J·δ ≈ 0  iff  Z(V^T δ) = 0  iff  V^T δ = 0  (when rank(Z)=K).

Projection formula (same for both modes):
    delta_proj = null_basis.T @ (null_basis @ delta_flat)
"""

import logging
import torch
from typing import Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Classic mode — for small matrices / unit tests
# ---------------------------------------------------------------------------

def compute_null_space(
    J: torch.Tensor,
    threshold: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the null space of matrix J via full SVD.

    Use only for small matrices (unit tests, tiny mel crops).
    For real audio use compute_null_space_factored().

    Parameters
    ----------
    J : torch.Tensor, shape [output_dim, input_dim]
    threshold : float  — singular value cutoff

    Returns
    -------
    null_basis : torch.Tensor, shape [null_dim, input_dim]
    singular_values : torch.Tensor
    """
    logger.info("Computing null space via SVD on matrix of shape %s", list(J.shape))
    J_f = J.float()
    U, S, Vh = torch.linalg.svd(J_f, full_matrices=True)

    output_dim, input_dim = J.shape
    rank = int((S > threshold).sum().item())
    null_dim = input_dim - rank

    logger.info(
        "SVD complete | input_dim=%d | output_dim=%d | "
        "effective_rank=%d | null_dim=%d | null_fraction=%.1f%%",
        input_dim, output_dim, rank, null_dim,
        100.0 * null_dim / input_dim if input_dim > 0 else 0.0,
    )
    print(
        f"[NullSpace] effective rank = {rank} / {input_dim}  "
        f"| null dim = {null_dim}  "
        f"| {100.0 * null_dim / input_dim:.1f}% of input space is null"
    )

    if null_dim == 0:
        logger.warning("Null space is empty — returning empty basis.")
        null_basis = torch.zeros(0, input_dim, dtype=J_f.dtype, device=J.device)
    else:
        null_basis = Vh[rank:, :]          # [null_dim, input_dim]

    return null_basis, S


# ---------------------------------------------------------------------------
# 2. Factored mode — SCALABLE for real audio
# ---------------------------------------------------------------------------

def compute_null_space_factored(
    Z: torch.Tensor,
    V: torch.Tensor,
    threshold: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract null space from the FACTORED Jacobian J ≈ (1/K) Z @ V^T.

    This avoids materialising the full [N x N] matrix.
    SVD is performed on V.T [K, N] — cost O(K^2 * N) instead of O(N^3).

    Parameters
    ----------
    Z : torch.Tensor, shape [input_dim, K]
        JVP vectors (columns = J·v_k).
    V : torch.Tensor, shape [input_dim, K]
        Probe vectors.
    threshold : float
        Singular value cutoff on V.T.

    Returns
    -------
    null_basis : torch.Tensor, shape [null_dim, input_dim]
        Orthonormal null space basis.  null_dim ≈ input_dim - K.
    singular_values : torch.Tensor, shape [K]
        Singular values of V.T (for diagnostics).
    """
    input_dim, K = V.shape
    logger.info(
        "Factored null space | input_dim=%d | K=%d", input_dim, K
    )

    # SVD of V^T  [K x input_dim]
    # COMPACT form (full_matrices=False): U [K,K], S [K], Vh [K, input_dim]
    # Avoids allocating the full [input_dim, input_dim] Vh matrix
    # (which would be ~N^2 * 4 bytes, e.g. 9.3 GB for a 9-second clip).
    #
    # The K rows of Vh form an orthonormal basis for row(V^T) = col(V).
    # Projection onto null(V^T) uses the COMPLEMENT formula:
    #   P_null(delta) = delta - Vh.T @ (Vh @ delta)
    # This is equivalent to the full-matrix approach but O(K*N) instead of O(N^2).
    Vt = V.T.float()                         # [K, input_dim]
    U, S, Vh = torch.linalg.svd(Vt, full_matrices=False)  # Vh: [K, input_dim]

    # Count how many singular values of V^T exceed threshold
    rank_Vt = int((S > threshold).sum().item())
    null_dim = input_dim - rank_Vt           # ≈ input_dim - K

    pct = 100.0 * null_dim / input_dim if input_dim > 0 else 0.0
    print(
        f"[NullSpace-factored] K={K} probes | rank(V^T)={rank_Vt} | "
        f"null_dim={null_dim} | {pct:.1f}% of input space is null"
    )
    logger.info(
        "Factored SVD done | rank(V^T)=%d | null_dim=%d | null_frac=%.1f%%",
        rank_Vt, null_dim, pct,
    )

    if null_dim == 0:
        logger.warning("Factored null space is empty.")
        # Return a zero-row range basis; project_onto_null_space will catch shape[0]==0
        range_basis = torch.zeros(0, input_dim, dtype=torch.float32)
    else:
        # range_basis [rank_Vt, input_dim]: orthonormal basis of col(V)
        # Stored compactly; projection uses complement: P_null = I - range.T @ range
        range_basis = Vh[:rank_Vt, :]        # [rank_Vt, input_dim]

    # We return range_basis in the slot previously called null_basis.
    # project_onto_null_space detects the compact form (shape[0] < shape[1])
    # and automatically uses the complement projection formula.
    return range_basis, S


# ---------------------------------------------------------------------------
# 3. Shared projection — works with basis from either mode
# ---------------------------------------------------------------------------

def project_onto_null_space(
    delta: torch.Tensor,
    null_basis: torch.Tensor,
) -> torch.Tensor:
    """
    Project perturbation delta onto the null space.

    Formula:  delta_proj = null_basis.T @ (null_basis @ delta_flat)

    Parameters
    ----------
    delta : torch.Tensor  — any shape; flattened internally
    null_basis : torch.Tensor, shape [null_dim, input_dim]

    Returns
    -------
    delta_proj : torch.Tensor — same shape as delta
    """
    if null_basis.shape[0] == 0:
        logger.warning("Null basis is empty — returning zero perturbation.")
        return torch.zeros_like(delta)

    original_shape = delta.shape
    original_device = delta.device

    # Always project on CPU — matrices can be large for long audio.
    delta_flat = delta.reshape(-1).float().cpu()
    nb = null_basis.cpu().float()   # already CPU after SVD; explicit for safety

    if nb.shape[0] < nb.shape[1]:
        # Compact RANGE BASIS [K, input_dim] returned by compute_null_space_factored.
        # Null projection = complement of range: P_null = I - nb.T @ nb
        # Cost: O(K * N) — 1000x cheaper than the full null-basis approach.
        coords          = nb @ delta_flat           # [K]
        delta_proj_flat = delta_flat - nb.T @ coords  # [input_dim]
    else:
        # Full null basis [null_dim, input_dim] (unit tests / classic mode).
        # Project: keep only the component in the null space.
        coords          = nb @ delta_flat           # [null_dim]
        delta_proj_flat = nb.T @ coords             # [input_dim]

    delta_proj = delta_proj_flat.reshape(original_shape).to(dtype=delta.dtype, device=original_device)

    logger.debug(
        "Projection: ||delta||=%.4f  ||delta_proj||=%.4f  retained=%.1f%%",
        delta.norm().item(),
        delta_proj.norm().item(),
        100.0 * delta_proj.norm().item() / (delta.norm().item() + 1e-12),
    )
    return delta_proj


# ---------------------------------------------------------------------------
# 4. Verification helpers
# ---------------------------------------------------------------------------

def verify_null_space(
    J: torch.Tensor,
    null_basis: torch.Tensor,
    tolerance: float = 1e-4,
) -> float:
    """
    Verify J @ null_basis.T ≈ 0  (for the classic / exact Jacobian).
    Returns max residual.
    """
    if null_basis.shape[0] == 0:
        logger.warning("Cannot verify empty null basis.")
        return 0.0

    residual = J.float() @ null_basis.T.float()
    max_res = residual.abs().max().item()

    status = "PASS" if max_res < tolerance else "WARN"
    print(
        f"[NullSpace verify] max |J * null_basis.T| = {max_res:.2e}  "
        f"[tolerance={tolerance:.2e}]  [{status}]"
    )
    if max_res >= tolerance:
        logger.warning(
            "Verification residual %.2e exceeds tolerance %.2e.",
            max_res, tolerance,
        )
    return max_res


def verify_null_space_factored(
    V: torch.Tensor,
    null_basis: torch.Tensor,
    tolerance: float = 1e-3,
) -> float:
    """
    Verify that the null-space projection is correct.

    Supports both representations:
    - Full null basis [null_dim, N]: checks V.T @ null_basis.T ≈ 0
    - Compact range basis [K, N]:   projects a random vector into null(V^T)
                                    and checks V.T @ projection ≈ 0

    Returns max residual.
    """
    if null_basis.shape[0] == 0:
        logger.warning("Cannot verify empty null basis.")
        return 0.0

    nb = null_basis.float().cpu()
    Vt = V.T.float().cpu()   # [K, N]

    if nb.shape[0] < nb.shape[1]:
        # Compact range basis — verify via complement projection on a test vector.
        # Use float64 to reduce floating-point error in the two-step multiply.
        nb64 = nb.double()
        Vt64 = Vt.double()
        torch.manual_seed(0)
        test = torch.randn(nb.shape[1], dtype=torch.float64)  # random [N]
        coords = nb64 @ test                              # [K]
        proj   = test - nb64.T @ coords                  # null-space projection
        residual = Vt64 @ proj                            # should be ≈ 0
    else:
        # Full null basis — direct check: V.T @ null_basis.T ≈ 0
        residual = Vt @ nb.T                              # [K, null_dim]

    max_res = residual.abs().max().item()

    status = "PASS" if max_res < tolerance else "WARN"
    print(
        f"[NullSpace-factored verify] max |V^T * null.T| = {max_res:.2e}  "
        f"[tolerance={tolerance:.2e}]  [{status}]"
    )
    if max_res >= tolerance:
        logger.warning(
            "Factored verification residual %.2e exceeds tolerance %.2e.",
            max_res, tolerance,
        )
    return max_res
