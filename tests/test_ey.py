"""The Eckart-Young loss shared by the gradient and basis-expansion models."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime, minimize

from cca_zoo._utils._ey import (
    ey_cross_covariance,
    ey_grad_z,
    ey_loss,
    full_rank_reparametrisation,
    penalised_basis_ey_closed_form,
)


@pytest.mark.parametrize("n_views", [2, 3, 4])
def test_gradient_matches_finite_differences(n_views: int) -> None:
    """ey_grad_z is the derivative of ey_loss."""
    rng = np.random.default_rng(0)
    z = [rng.standard_normal((15, 3)) for _ in range(n_views)]

    def loss(flat: np.ndarray) -> float:
        return ey_loss(list(flat.reshape(n_views, 15, 3)))["objective"]

    np.testing.assert_allclose(
        np.ravel(ey_grad_z(z)), approx_fprime(np.ravel(z), loss, 1e-6), atol=1e-5
    )


def test_two_view_covariances() -> None:
    """V averages the views' covariances; C adds their cross-covariance."""
    rng = np.random.default_rng(0)
    z1, z2 = rng.standard_normal((30, 2)), rng.standard_normal((30, 2))
    v11, v22, c12 = (
        np.cov(a.T, b.T)[:2, 2:] for a, b in [(z1, z1), (z2, z2), (z1, z2)]
    )
    C, V = ey_cross_covariance([z1, z2])
    np.testing.assert_allclose(V, (v11 + v22) / 2, atol=1e-10)
    np.testing.assert_allclose(C, (v11 + v22 + c12 + c12.T) / 2, atol=1e-10)


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("difference_penalty", [False, True])
def test_closed_form_is_the_penalised_optimum(k: int, difference_penalty: bool) -> None:
    """The eigenproblem attains the best L-BFGS optimum, ridge or spline penalty."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 2))
    bases = [
        z @ rng.standard_normal((2, d)) + rng.standard_normal((200, d))
        for d in (6, 3, 4)
    ]
    bases = [b - b.mean(axis=0) for b in bases]
    dims = [b.shape[1] for b in bases]
    shapes = [
        np.diff(np.eye(d), n=2, axis=0).T @ np.diff(np.eye(d), n=2, axis=0)
        if difference_penalty
        else np.eye(d)
        for d in dims
    ]
    penalties = [s * p for s, p in zip((0.1, 2.0, 0.0), shapes)]
    split = np.cumsum([d * k for d in dims])[:-1]

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        coefs = [c.reshape(d, k) for c, d in zip(np.split(x, split), dims)]
        scores = [b @ c for b, c in zip(bases, coefs)]
        loss = ey_loss(scores)["objective"] + 0.5 * sum(
            float(np.sum(c * (p @ c))) for c, p in zip(coefs, penalties)
        )
        grads = [
            b.T @ g + p @ c
            for b, g, c, p in zip(bases, ey_grad_z(scores), coefs, penalties)
        ]
        return loss, np.concatenate([g.ravel() for g in grads])

    closed = objective(
        np.concatenate(
            [c.ravel() for c in penalised_basis_ey_closed_form(bases, k, penalties)]
        )
    )[0]
    iterative = min(
        minimize(
            objective,
            rng.standard_normal(split[-1] + dims[-1] * k),
            jac=True,
            method="L-BFGS-B",
            options={"gtol": 1e-12},
        ).fun
        for _ in range(5)
    )
    np.testing.assert_allclose(closed, iterative, rtol=1e-6)


def test_reparametrisation_is_full_rank_and_keeps_the_penalty() -> None:
    """A collinear basis becomes full rank with the minimum-penalty coefficients."""
    rng = np.random.default_rng(0)
    raw = rng.random((100, 8))
    raw[:, -1] = 0.0
    raw[:, :4] /= raw[:, :4].sum(axis=1, keepdims=True)
    basis = raw - raw.mean(axis=0)
    factor = np.diff(np.eye(8), n=2, axis=0)
    lift, _, reduced_penalty = full_rank_reparametrisation(
        basis, basis.T @ basis, factor, np.linalg.norm(factor, 2)
    )
    rank = np.linalg.matrix_rank(basis)
    assert lift.shape[1] == rank == np.linalg.matrix_rank(basis @ lift)
    a = rng.standard_normal(rank)
    w = lift @ a
    np.testing.assert_allclose(
        a @ reduced_penalty @ a, np.sum((factor @ w) ** 2), rtol=1e-10
    )
    null = np.linalg.svd(basis)[2][rank:].T
    others = w[:, None] + null @ rng.standard_normal((null.shape[1], 20))
    assert np.all(
        np.sum((factor @ others) ** 2, axis=0) >= np.sum((factor @ w) ** 2) - 1e-9
    )
