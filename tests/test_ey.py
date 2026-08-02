"""Tests for cca_zoo._utils._ey (shared Eckart-Young loss machinery)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._utils._ey import ey_cross_covariance, ey_grad_z, ey_loss


def _numerical_grad_z(
    representations: list[np.ndarray], eps: float = 1e-6
) -> list[np.ndarray]:
    """Central-difference numerical gradient of ey_loss w.r.t. each Z_i."""
    grads = []
    for idx, z in enumerate(representations):
        g = np.zeros_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                plus = [r.copy() for r in representations]
                minus = [r.copy() for r in representations]
                plus[idx][i, j] += eps
                minus[idx][i, j] -= eps
                g[i, j] = (ey_loss(plus)["objective"] - ey_loss(minus)["objective"]) / (
                    2 * eps
                )
        grads.append(g)
    return grads


@pytest.mark.parametrize("n_views", [2, 3, 4])
def test_ey_grad_z_matches_finite_difference(n_views: int) -> None:
    """ey_grad_z matches a numerical gradient of ey_loss for M = 2, 3, 4 views."""
    rng = np.random.default_rng(0)
    n, k = 15, 3
    representations = [rng.standard_normal((n, k)) for _ in range(n_views)]
    analytic = ey_grad_z(representations)
    numeric = _numerical_grad_z(representations)
    for a, b in zip(analytic, numeric):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_ey_cross_covariance_shapes() -> None:
    """ey_cross_covariance returns (k, k) matrices for C and V."""
    rng = np.random.default_rng(0)
    representations = [rng.standard_normal((20, 4)) for _ in range(3)]
    C, V = ey_cross_covariance(representations)
    assert C.shape == (4, 4)
    assert V.shape == (4, 4)


def test_ey_cross_covariance_two_views_matches_manual() -> None:
    """For M=2, C and V match a manual pairwise-covariance computation."""
    rng = np.random.default_rng(0)
    n = 30
    z1 = rng.standard_normal((n, 2))
    z2 = rng.standard_normal((n, 2))
    z1c = z1 - z1.mean(axis=0)
    z2c = z2 - z2.mean(axis=0)
    v11 = z1c.T @ z1c / (n - 1)
    v22 = z2c.T @ z2c / (n - 1)
    c12 = z1c.T @ z2c / (n - 1)
    expected_V = (v11 + v22) / 2
    expected_C = (v11 + v22 + c12 + c12.T) / 2

    C, V = ey_cross_covariance([z1, z2])
    np.testing.assert_allclose(V, expected_V, atol=1e-10)
    np.testing.assert_allclose(C, expected_C, atol=1e-10)


def test_ey_loss_zero_for_zero_embeddings() -> None:
    """The EY loss is exactly zero when all embeddings are exactly zero."""
    representations = [np.zeros((10, 2)), np.zeros((10, 2))]
    result = ey_loss(representations)
    assert result["objective"] == pytest.approx(0.0)
    assert result["rewards"] == pytest.approx(0.0)
    assert result["penalties"] == pytest.approx(0.0)


def test_ey_loss_perfectly_correlated_views() -> None:
    """Perfectly correlated, unit-variance views give a large negative loss."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((100, 1))
    z = z / z.std()
    result = ey_loss([z, z])
    # C == V here (both views identical), so objective = -2*tr(V) + tr(V@V).
    assert result["objective"] < 0.0
