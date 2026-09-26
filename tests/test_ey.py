"""Tests for cca_zoo._utils._ey (shared Eckart-Young loss machinery)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._utils._ey import (
    ey_cross_covariance,
    ey_grad_z,
    ey_loss,
    full_rank_reparametrisation,
    penalised_basis_ey_closed_form,
)


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


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("matrix_penalty", [False, True])
def test_penalised_closed_form_matches_iterative_optimum(
    k: int, matrix_penalty: bool
) -> None:
    """The eigenproblem solution attains the best L-BFGS optimum from many starts.

    Three views of unequal width with per-view penalties — a ridge (one of
    them zero), or a second-difference penalty matrix as GAMCCA uses.
    """
    import scipy.optimize

    rng = np.random.default_rng(0)
    n = 200
    z = rng.standard_normal((n, 2))
    bases = [
        z @ rng.standard_normal((2, d)) + rng.standard_normal((n, d)) for d in (6, 3, 4)
    ]
    bases = [b - b.mean(axis=0) for b in bases]
    dims = [b.shape[1] for b in bases]
    if matrix_penalty:
        diffs = [np.diff(np.eye(d), n=2, axis=0) for d in dims]
        penalties = [s * (dd.T @ dd) for s, dd in zip((0.1, 2.0, 0.0), diffs)]
    else:
        penalties = [r * np.eye(d) for r, d in zip((0.1, 2.0, 0.0), dims)]
    split = np.cumsum([d * k for d in dims])[:-1]

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        coefs = [c.reshape(d, k) for c, d in zip(np.split(x, split), dims)]
        reps = [b @ c for b, c in zip(bases, coefs)]
        loss = ey_loss(reps)["objective"] + 0.5 * sum(
            float(np.sum(c * (p @ c))) for c, p in zip(coefs, penalties)
        )
        grads = [
            b.T @ g + p @ c
            for b, g, c, p in zip(bases, ey_grad_z(reps), coefs, penalties)
        ]
        return loss, np.concatenate([g.ravel() for g in grads])

    closed = penalised_basis_ey_closed_form(bases, k, penalties)
    closed_loss = objective(np.concatenate([c.ravel() for c in closed]))[0]
    iterative_loss = min(
        scipy.optimize.minimize(
            objective,
            rng.standard_normal(sum(d * k for d in dims)),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5000, "gtol": 1e-12},
        ).fun
        for _ in range(5)
    )
    assert [c.shape for c in closed] == [(d, k) for d in dims]
    assert closed_loss <= iterative_loss + 1e-8
    np.testing.assert_allclose(closed_loss, iterative_loss, rtol=1e-6)


def test_full_rank_reparametrisation_is_exact() -> None:
    """The reduced problem reproduces the original's embeddings and penalties.

    A P-spline-like basis with an exactly collinear block (a partition of
    unity, centred) and a column with no data: every reduced coefficient
    lifts to original coefficients giving the same embedding, and the
    reduced penalty is the smallest any such coefficients attain.
    """
    rng = np.random.default_rng(0)
    n, d = 100, 8
    raw = rng.random((n, d))
    raw[:, -1] = 0.0  # a spline with no data under it
    raw[:, :4] /= raw[:, :4].sum(axis=1, keepdims=True)  # partition of unity
    basis = raw - raw.mean(axis=0)
    factor = np.diff(np.eye(d), n=2, axis=0)
    reduced, reduced_penalty, lift = full_rank_reparametrisation(basis, factor)
    assert reduced.shape[1] == np.linalg.matrix_rank(basis)
    a = rng.standard_normal(reduced.shape[1])
    w = lift @ a
    np.testing.assert_allclose(basis @ w, reduced @ a, atol=1e-10)
    np.testing.assert_allclose(
        a @ reduced_penalty @ a, np.sum((factor @ w) ** 2), rtol=1e-10
    )
    # No other coefficients with the same embedding have a smaller penalty.
    null = np.linalg.svd(basis)[2][reduced.shape[1] :].T
    for _ in range(20):
        other = w + null @ rng.standard_normal(null.shape[1])
        assert np.sum((factor @ other) ** 2) >= np.sum((factor @ w) ** 2) - 1e-9


def test_reparametrisation_ignores_directions_the_penalty_annihilates() -> None:
    """A null space the penalty also annihilates gets no coefficient at all.

    Two features' B-spline blocks each sum to one, so centring makes every
    feature's constant a null direction of the basis, and a difference
    penalty annihilates constants too. The minimum-norm solution puts
    nothing there; numerically those directions have rounding-sized
    penalty singular values, which a cutoff relative to the matrix itself
    (rather than to the penalty) inverted into coefficients of ~1e15 —
    harmless to the embedding but ruinous to each feature's own term.
    """
    rng = np.random.default_rng(0)
    n, k = 400, 10
    blocks = []
    for _ in range(2):
        raw = rng.random((n, k)) ** 3
        blocks.append(raw / raw.sum(axis=1, keepdims=True))
    basis = np.hstack(blocks)
    basis -= basis.mean(axis=0)
    first_differences = np.diff(np.eye(k), n=1, axis=0)
    factor = np.sqrt(1e-3) * np.kron(np.eye(2), first_differences)
    reduced, _, lift = full_rank_reparametrisation(basis, factor)
    a = rng.standard_normal(reduced.shape[1])
    w = lift @ a
    np.testing.assert_allclose(basis @ w, reduced @ a, atol=1e-10)
    per_feature = [basis[:, :k] @ w[:k], basis[:, k:] @ w[k:]]
    np.testing.assert_allclose(sum(per_feature), reduced @ a, atol=1e-8)
    assert np.abs(w).max() < 1e3 * np.abs(a).max()
