"""Tests for cca_zoo.linear.gradient._svcca (bounded-influence EY loss)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._utils._ey import ey_grad_z, ey_loss
from cca_zoo.linear import CCAEY, SupportVectorCCA
from cca_zoo.linear.gradient._svcca import _huber_sample_weight, _weighted_ey


def _numerical_grad_z(
    representations: list[np.ndarray], sample_weight: np.ndarray, eps: float = 1e-6
) -> list[np.ndarray]:
    """Central-difference numerical gradient of _weighted_ey w.r.t. each Z_i.

    ``sample_weight`` is held fixed across the perturbation, matching how
    ``_weighted_ey`` treats it as given rather than differentiating through
    it -- the same stop-gradient IRLS gives its weights.
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
                obj_p, _ = _weighted_ey(plus, sample_weight)
                obj_m, _ = _weighted_ey(minus, sample_weight)
                g[i, j] = (obj_p - obj_m) / (2 * eps)
        grads.append(g)
    return grads


@pytest.mark.parametrize("n_views", [2, 3, 4])
def test_weighted_ey_grad_matches_finite_difference(n_views: int) -> None:
    """_weighted_ey's gradient matches a numerical gradient, for fixed weights."""
    rng = np.random.default_rng(0)
    n, k = 15, 3
    representations = [rng.standard_normal((n, k)) for _ in range(n_views)]
    sample_weight = rng.uniform(0.2, 1.0, size=n)
    _, analytic = _weighted_ey(representations, sample_weight)
    numeric = _numerical_grad_z(representations, sample_weight)
    for a, b in zip(analytic, numeric):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_weighted_ey_reduces_to_plain_ey_at_uniform_weight() -> None:
    """Uniform sample_weight=1 reproduces ey_loss/ey_grad_z exactly."""
    rng = np.random.default_rng(0)
    representations = [rng.standard_normal((20, 2)) for _ in range(2)]
    uniform = np.ones(20)
    objective, grad = _weighted_ey(representations, uniform)
    expected_objective = ey_loss(representations)["objective"]
    expected_grad = ey_grad_z(representations)
    assert objective == pytest.approx(expected_objective, abs=1e-10)
    for a, b in zip(grad, expected_grad):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_huber_sample_weight_keeps_at_least_half_the_batch_at_one() -> None:
    """Delta >= 1 guarantees at least half the batch keeps weight 1.

    The cutoff is delta times the batch's own median leverage, so it never
    starves the effective sample size regardless of batch size or
    dimensionality (unlike a fixed absolute leverage radius).
    """
    rng = np.random.default_rng(0)
    representations = [rng.standard_normal((40, 2)) for _ in range(2)]
    weight = _huber_sample_weight(representations, delta=1.0)
    assert (weight >= 1.0 - 1e-9).sum() >= 20


def test_robust_to_high_leverage_outliers_unlike_ccaey() -> None:
    """A few spurious high-leverage training points wreck CCAEY but not this.

    Two views share a real latent factor (true achievable correlation is
    high); a small fraction of training points are replaced by a spurious,
    near-perfectly-correlated cluster along a different, large-magnitude
    direction -- the classic high-leverage contamination that dominates a
    covariance-based objective. Measured on a clean, uncontaminated test
    set, so a model that got pulled towards the spurious cluster in training
    generalises poorly.
    """
    rng = np.random.default_rng(0)
    n_train, n_test, p1, p2 = 400, 300, 8, 6
    w1 = rng.standard_normal(p1)
    w1 /= np.linalg.norm(w1)
    w2 = rng.standard_normal(p2)
    w2 /= np.linalg.norm(w2)
    u1 = rng.standard_normal(p1)
    u1 /= np.linalg.norm(u1)
    u2 = rng.standard_normal(p2)
    u2 /= np.linalg.norm(u2)

    def clean(n: int) -> tuple[np.ndarray, np.ndarray]:
        t = rng.standard_normal(n)
        X = np.outer(t, w1) + 0.6 * rng.standard_normal((n, p1))
        Y = np.outer(t, w2) + 0.6 * rng.standard_normal((n, p2))
        return X, Y

    x_test, y_test = clean(n_test)
    x_train, y_train = clean(n_train)

    n_out = int(round(0.05 * n_train))
    idx = rng.choice(n_train, n_out, replace=False)
    s = rng.standard_normal(n_out)
    x_train[idx] = np.outer(s, u1) * 9.0
    y_train[idx] = np.outer(s, u2) * 9.0

    def held_out_corr(model: CCAEY | SupportVectorCCA) -> float:
        z1, z2 = model.transform([x_test, y_test])
        return abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])

    kwargs = dict(latent_dimensions=1, max_iter=1500, random_state=0)
    ccaey_corr = held_out_corr(CCAEY(**kwargs).fit([x_train, y_train]))
    svcca_corr = held_out_corr(SupportVectorCCA(**kwargs).fit([x_train, y_train]))

    assert svcca_corr > ccaey_corr + 0.2
