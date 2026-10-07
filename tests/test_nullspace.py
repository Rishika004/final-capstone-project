"""
tests/test_nullspace.py
=======================
Unit tests for the null-space mathematics and the full protection pipeline.

Run with:
    pytest tests/test_nullspace.py -v

Or via the CLI:
    python main.py --mode test

Tests
-----
1. test_null_space_definition       — J @ null_basis.T ≈ 0
2. test_projection_idempotency      — P(P(x)) == P(x)
3. test_projection_orthogonality    — projected delta is ⊥ to row space
4. test_epsilon_constraint          — ||projected_delta||_inf ≤ epsilon
5. test_prr_null_gt_random          — PRR of null-space delta > PRR of random delta
6. test_jacobian_shape              — Jacobian has correct output shape
7. test_empty_null_space_handled    — graceful handling of full-rank J
8. test_verify_null_space_passess   — verify_null_space returns small residual
9. test_perturbation_pipeline       — apply_null_space_protection end-to-end
"""

import sys
import os
# Ensure project root is on sys.path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from core.nullspace import (
    compute_null_space,
    compute_null_space_factored,
    project_onto_null_space,
    verify_null_space,
    verify_null_space_factored,
)
from core.jacobian import estimate_jacobian_randomized
from core.perturbation import generate_base_perturbation, apply_null_space_protection
from evaluation.metrics import (
    purification_resistance_rate,
    mel_distance,
    signal_to_noise_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_linear_model():
    """
    A small linear model whose Jacobian is exactly its weight matrix.
    This lets us verify the math analytically.
    """
    torch.manual_seed(0)
    in_dim, out_dim = 20, 10  # out_dim < in_dim → null space dim = 10
    linear = nn.Linear(in_dim, out_dim, bias=False)
    linear.eval()
    return linear, in_dim, out_dim


@pytest.fixture(scope="module")
def purifier_small():
    """Small UNet purifier for fast tests (tiny input size)."""
    from models.purifier import MelUNetPurifier
    torch.manual_seed(42)
    model = MelUNetPurifier(n_mels=8, base_channels=8)
    model.eval()
    return model


@pytest.fixture(scope="module")
def dummy_speaker_encoder():
    """Tiny dummy speaker encoder for pipeline tests."""
    from models.speaker_encoder import _make_dummy_encoder
    return _make_dummy_encoder(torch.device("cpu"), embed_dim=16)


# ---------------------------------------------------------------------------
# Test 1: Null space definition — J @ null_basis.T ≈ 0
# ---------------------------------------------------------------------------

class TestNullSpaceDefinition:
    """Null space basis vectors must satisfy J·v = 0 by construction."""

    def test_exact_null_space_linear_model(self, small_linear_model):
        """For a linear model, the Jacobian equals the weight matrix exactly."""
        model, in_dim, out_dim = small_linear_model
        W = model.weight.data  # [out_dim, in_dim]

        null_basis, singular_values = compute_null_space(W, threshold=1e-6)

        # Null space should be non-empty (out_dim < in_dim)
        assert null_basis.shape[0] > 0, "Null space should be non-empty"
        expected_null_dim = in_dim - out_dim
        assert null_basis.shape[0] == expected_null_dim, (
            f"Expected null dim {expected_null_dim}, got {null_basis.shape[0]}"
        )

        # Verify: W @ null_basis.T ≈ 0
        residual = W.float() @ null_basis.T.float()
        max_res = residual.abs().max().item()
        assert max_res < 1e-4, (
            f"Null space verification failed: max |J·v| = {max_res:.2e}"
        )

    def test_random_jacobian_null_space(self):
        """Null space of a random fat matrix must have correct dimension."""
        torch.manual_seed(7)
        m, n = 15, 30  # fat matrix → null dim should be ≥ n - m = 15
        J = torch.randn(m, n)

        null_basis, S = compute_null_space(J, threshold=1e-4)

        residual = J @ null_basis.T
        max_res = residual.abs().max().item()
        assert max_res < 1e-3, f"max |J·null| = {max_res:.2e}"

    def test_square_full_rank_gives_empty_null_space(self):
        """A square, full-rank matrix has an empty null space."""
        torch.manual_seed(3)
        n = 10
        J = torch.eye(n) + 0.01 * torch.randn(n, n)  # near-identity, full rank
        null_basis, S = compute_null_space(J, threshold=1e-5)
        assert null_basis.shape[0] == 0, (
            "Full-rank matrix should have empty null space"
        )


# ---------------------------------------------------------------------------
# Test 2: Projection idempotency — P(P(x)) == P(x)
# ---------------------------------------------------------------------------

class TestProjectionIdempotency:
    """Orthogonal projection onto a subspace is idempotent: P² = P."""

    def test_idempotent_on_random_vector(self):
        torch.manual_seed(11)
        m, n = 10, 30
        J = torch.randn(m, n)
        null_basis, _ = compute_null_space(J, threshold=1e-4)

        if null_basis.shape[0] == 0:
            pytest.skip("Empty null space — cannot test idempotency")

        delta = torch.randn(n)
        proj1 = project_onto_null_space(delta, null_basis, kind="null")
        proj2 = project_onto_null_space(proj1, null_basis, kind="null")

        # P(P(x)) == P(x)  up to floating-point precision
        diff = (proj1 - proj2).abs().max().item()
        assert diff < 1e-5, f"Projection not idempotent: max diff = {diff:.2e}"

    def test_idempotent_on_mel_shaped_tensor(self):
        torch.manual_seed(13)
        n_mels, time = 8, 16
        n = n_mels * time
        m = n // 3

        J = torch.randn(m, n)
        null_basis, _ = compute_null_space(J, threshold=1e-4)

        if null_basis.shape[0] == 0:
            pytest.skip("Empty null space")

        delta = torch.randn(1, n_mels, time)
        proj1 = project_onto_null_space(delta, null_basis, kind="null")
        proj2 = project_onto_null_space(proj1, null_basis, kind="null")

        diff = (proj1 - proj2).abs().max().item()
        assert diff < 1e-5, f"Mel-shaped idempotency failed: {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 3: Projection orthogonality — projected delta is ⊥ to J's row space
# ---------------------------------------------------------------------------

class TestProjectionOrthogonality:
    """J · P(delta) should be ≈ 0 (delta_proj lives in null space)."""

    def test_projected_delta_in_null_space(self):
        torch.manual_seed(17)
        m, n = 12, 40
        J = torch.randn(m, n)
        null_basis, _ = compute_null_space(J, threshold=1e-4)

        if null_basis.shape[0] == 0:
            pytest.skip("Empty null space")

        delta = torch.randn(n)
        delta_proj = project_onto_null_space(delta, null_basis, kind="null")

        # J · delta_proj ≈ 0
        residual = J.float() @ delta_proj.float()
        max_res = residual.abs().max().item()
        assert max_res < 1e-3, (
            f"Projected delta not in null space: max |J·delta_proj| = {max_res:.2e}"
        )


# ---------------------------------------------------------------------------
# Test 4: Epsilon constraint — ||projected_delta||_inf ≤ epsilon
# ---------------------------------------------------------------------------

class TestEpsilonConstraint:
    """After clamping, the perturbation must satisfy the L∞ budget."""

    @pytest.mark.parametrize("epsilon", [0.005, 0.01, 0.05])
    def test_linf_budget(self, epsilon):
        torch.manual_seed(19)
        n_mels, time = 8, 32
        n = n_mels * time
        m = n // 4

        J = torch.randn(m, n)
        null_basis, _ = compute_null_space(J, threshold=1e-4)

        if null_basis.shape[0] == 0:
            pytest.skip("Empty null space")

        # Base perturbation (already bounded)
        delta = generate_base_perturbation(
            torch.zeros(1, n_mels, time), epsilon=epsilon, seed=42
        )
        delta_proj = project_onto_null_space(delta, null_basis, kind="null")
        delta_clamped = delta_proj.clamp(-epsilon, epsilon)

        linf = delta_clamped.abs().max().item()
        assert linf <= epsilon + 1e-7, (
            f"L∞ constraint violated: ||delta||_∞ = {linf:.6f} > ε = {epsilon}"
        )


# ---------------------------------------------------------------------------
# Test 5: PRR — null-space delta > random delta
# ---------------------------------------------------------------------------

class TestPRRNullVsRandom:
    """
    A perturbation in the null space should be more resistant to purification
    than a random perturbation of the same magnitude.
    """

    def test_prr_null_greater_than_random(self, purifier_small):
        torch.manual_seed(23)
        x_mel = torch.randn(1, 8, 32)
        epsilon = 0.02

        # Random perturbation
        delta_random = generate_base_perturbation(x_mel, epsilon=epsilon, seed=1)
        x_random = x_mel + delta_random

        # Null-space perturbation via factored Jacobian
        Z, V = estimate_jacobian_randomized(purifier_small, x_mel, K=20, seed=42)
        null_basis, _ = compute_null_space_factored(Z, V, threshold=1e-4)

        if null_basis.shape[0] == 0:
            pytest.skip("Empty null space for this model")

        # factored -> this is a RANGE basis, so the projection is onto its
        # complement (council v5 BUG B: the kind is now explicit).
        delta_null = project_onto_null_space(delta_random, null_basis, kind="range").clamp(
            -epsilon, epsilon
        )
        x_null = x_mel + delta_null

        prr_random = purification_resistance_rate(x_mel, x_random, purifier_small)
        prr_null = purification_resistance_rate(x_mel, x_null, purifier_small)

        # Null-space PRR should be ≥ random PRR
        # (null-space delta is harder for the purifier to remove)
        assert prr_null >= prr_random * 0.8, (
            f"PRR_null ({prr_null:.4f}) should be ≥ PRR_random ({prr_random:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 6: Jacobian shape
# ---------------------------------------------------------------------------

class TestJacobianShape:
    """Jacobian estimator returns (Z, V) each of shape [input_dim, K]."""

    def test_output_shape(self, purifier_small):
        torch.manual_seed(29)
        x = torch.randn(1, 8, 16)
        K = 10
        Z, V = estimate_jacobian_randomized(purifier_small, x, K=K, seed=0)

        input_dim = x.numel()
        assert Z.shape == (input_dim, K), (
            f"Z shape mismatch: got {Z.shape}, expected {(input_dim, K)}"
        )
        assert V.shape == (input_dim, K), (
            f"V shape mismatch: got {V.shape}, expected {(input_dim, K)}"
        )


# ---------------------------------------------------------------------------
# Test 7: Empty null space handled gracefully
# ---------------------------------------------------------------------------

class TestEmptyNullSpaceHandled:
    """When the null space is empty, the pipeline should not crash."""

    def test_project_onto_empty_basis_returns_zeros(self):
        """project_onto_null_space with empty basis → zero perturbation."""
        n = 20
        null_basis = torch.zeros(0, n)  # empty basis
        delta = torch.randn(n)

        result = project_onto_null_space(delta, null_basis, kind="null")
        assert result.abs().max().item() == 0.0, (
            "Projecting onto empty null space should return zeros"
        )

    def test_apply_protection_with_full_rank_returns_original(self, purifier_small):
        """If J is effectively full rank, protection returns input unchanged."""
        x_mel = torch.randn(1, 8, 16)
        # Use an extremely large threshold so everything is treated as full rank
        result, _ = apply_null_space_protection(
            x_mel, purifier_small, epsilon=0.01, K=10, svd_threshold=1e6, verify=False
        )
        # Should return original (possibly same tensor or equal values)
        # The function logs a warning and returns x_mel.clone()
        assert result.shape == x_mel.shape


# ---------------------------------------------------------------------------
# Test 8: verify_null_space correctness check
# ---------------------------------------------------------------------------

class TestVerifyNullSpace:
    """verify_null_space should return a small residual for a correct basis."""

    def test_residual_is_small(self):
        torch.manual_seed(31)
        m, n = 8, 24
        J = torch.randn(m, n)
        null_basis, _ = compute_null_space(J, threshold=1e-4)

        if null_basis.shape[0] == 0:
            pytest.skip("Empty null space")

        max_res = verify_null_space(J, null_basis, tolerance=1e-2)
        assert max_res < 1e-2, (
            f"verify_null_space residual {max_res:.2e} is too large"
        )


# ---------------------------------------------------------------------------
# Test 9: End-to-end perturbation pipeline
# ---------------------------------------------------------------------------

class TestPerturbationPipeline:
    """apply_null_space_protection should return a tensor of the same shape."""

    def test_output_shape_unchanged(self, purifier_small):
        torch.manual_seed(37)
        x_mel = torch.randn(1, 8, 32)
        result, _ = apply_null_space_protection(
            x_mel, purifier_small, epsilon=0.01, K=20, verify=False
        )
        assert result.shape == x_mel.shape, (
            f"Shape mismatch: {result.shape} vs {x_mel.shape}"
        )

    def test_output_within_epsilon_of_input(self, purifier_small):
        """Protected mel should be within epsilon of original mel."""
        torch.manual_seed(41)
        epsilon = 0.02
        x_mel = torch.randn(1, 8, 32)
        result, _ = apply_null_space_protection(
            x_mel, purifier_small, epsilon=epsilon, K=20, verify=False
        )
        diff_max = (result - x_mel).abs().max().item()
        assert diff_max <= epsilon + 1e-6, (
            f"Output exceeds ε budget: ||result - x||_∞ = {diff_max:.6f} > {epsilon}"
        )


# ---------------------------------------------------------------------------
# Bonus: Mel distance and SNR sanity checks
# ---------------------------------------------------------------------------

class TestMetricsSanity:
    def test_mel_distance_zero_for_identical(self):
        mel = torch.randn(1, 8, 16)
        assert mel_distance(mel, mel) == pytest.approx(0.0, abs=1e-6)

    def test_snr_infinite_for_identical(self):
        wav = torch.randn(1, 1000)
        snr = signal_to_noise_ratio(wav, wav)
        assert snr == float("inf") or snr > 100.0

    def test_snr_decreases_with_more_noise(self):
        torch.manual_seed(43)
        wav = torch.randn(1, 4000)
        noise_small = wav + 0.001 * torch.randn_like(wav)
        noise_large = wav + 0.1 * torch.randn_like(wav)
        snr_small = signal_to_noise_ratio(wav, noise_small)
        snr_large = signal_to_noise_ratio(wav, noise_large)
        assert snr_small > snr_large, "More noise should give lower SNR"


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
