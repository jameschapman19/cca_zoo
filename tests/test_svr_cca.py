"""Tests for cca_zoo.linear.gradient._svr_cca (epsilon-insensitive EY loss)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import SupportVectorCCA
from cca_zoo.linear.gradient._svr_cca import _epsilon_insensitive_ey


def _numerical_grad_z(
    representations: list[np.ndarray],
    means: list[np.ndarray],
    stds: list[np.ndarray],
    epsilon: float,
    eps: float = 1e-6,
) -> list[np.ndarray]:
    """Central-difference numerical gradient of _epsilon_insensitive_ey.

    ``means``/``stds`` are held fixed across the perturbation, matching how
    the function treats them as given rather than differentiating through
    them -- the same stop-gradient convention as HuberCCA's sample weights.
    """
    grads = []
    for idx, z in enumerate(representations):
        g = np.zeros_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                plus = [r.copy() for r in representations]
                minus = [r.copy() for r in representations]
                plus[idx][i, j] += eps
                minus[idx][i, j] -= eps
                obj_p, _ = _epsilon_insensitive_ey(plus, means, stds, epsilon)
                obj_m, _ = _epsilon_insensitive_ey(minus, means, stds, epsilon)
                g[i, j] = (obj_p - obj_m) / (2 * eps)
        grads.append(g)
    return grads


@pytest.mark.parametrize("n_views", [2, 3, 4])
def test_epsilon_insensitive_ey_grad_matches_finite_difference(n_views: int) -> None:
    """The analytic gradient matches a numerical gradient, for fixed means/stds."""
    rng = np.random.default_rng(0)
    n, k = 20, 3
    representations = [
        rng.standard_normal((n, k)) * (1 + 0.3 * i) for i in range(n_views)
    ]
    means = [z.mean(axis=0) for z in representations]
    stds = [z.std(axis=0) + 1e-9 for z in representations]
    epsilon = 0.3
    _, analytic = _epsilon_insensitive_ey(representations, means, stds, epsilon)
    numeric = _numerical_grad_z(representations, means, stds, epsilon)
    for a, b in zip(analytic, numeric):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_zero_loss_and_gradient_when_within_the_tube() -> None:
    """Views already agreeing within epsilon give exactly zero loss and gradient.

    This is the genuine sparsity mechanism scikit-learn's SVR has and a
    Huber-style estimator does not: points inside the tube contribute
    *nothing*, not just something small.
    """
    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 2))
    representations = [z, z + 0.01 * rng.standard_normal((50, 2))]
    means = [r.mean(axis=0) for r in representations]
    stds = [r.std(axis=0) + 1e-9 for r in representations]
    objective, grad = _epsilon_insensitive_ey(representations, means, stds, epsilon=1.0)
    assert objective == pytest.approx(0.0)
    for g in grad:
        np.testing.assert_allclose(g, 0.0)


def test_gradient_is_exactly_sparse_across_samples() -> None:
    """Some sample-components get exactly zero gradient, others don't.

    Constructed so most samples already agree across views (within epsilon)
    while a few are deliberately misaligned -- the aligned ones should drop
    out of the gradient entirely.
    """
    rng = np.random.default_rng(0)
    n, k = 60, 1
    base = rng.standard_normal((n, k))
    z1 = base.copy()
    z2 = base.copy()
    misaligned = rng.choice(n, size=10, replace=False)
    z2[misaligned] += 5.0  # push a subset far from consensus
    means = [z1.mean(axis=0), z2.mean(axis=0)]
    stds = [z1.std(axis=0) + 1e-9, z2.std(axis=0) + 1e-9]
    _, grad = _epsilon_insensitive_ey([z1, z2], means, stds, epsilon=0.2)
    nonzero = np.abs(grad[1][:, 0]) > 1e-12
    assert nonzero.sum() < n  # not every sample contributes
    assert set(np.where(nonzero)[0]) >= set(misaligned)  # misaligned ones do


def test_fit_recovers_high_correlation_on_correlated_data(
    correlated_views: list[np.ndarray],
) -> None:
    """Fits a real shared-latent-factor dataset to a high canonical correlation."""
    model = SupportVectorCCA(
        latent_dimensions=1,
        epsilon=0.1,
        max_iter=2000,
        learning_rate=0.05,
        random_state=0,
    )
    model.fit(correlated_views)
    z1, z2 = model.transform(correlated_views)
    corr = abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])
    assert corr > 0.8
