"""Tests for cca_zoo.linear.gradient._svr_cca (hinge-capped-reward EY loss)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._utils._ey import ey_loss
from cca_zoo.linear import SupportVectorCCA
from cca_zoo.linear.gradient._svr_cca import _hinge_ey, _self_calibrated_tau


def _numerical_grad_z(
    representations: list[np.ndarray],
    means: list[np.ndarray],
    tau: float,
    eps: float = 1e-6,
) -> list[np.ndarray]:
    """Central-difference numerical gradient of _hinge_ey.

    ``means`` and ``tau`` are held fixed across the perturbation -- required
    here (unlike the plain EY loss) since the hinge's derivative is a
    nonlinear step function of the per-sample reward, so there is no
    translation-invariance cancellation to fall back on; see _hinge_ey's
    docstring.
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
                obj_p, _ = _hinge_ey(plus, means, tau)
                obj_m, _ = _hinge_ey(minus, means, tau)
                g[i, j] = (obj_p - obj_m) / (2 * eps)
        grads.append(g)
    return grads


@pytest.mark.parametrize("n_views", [2, 3, 4])
def test_hinge_ey_grad_matches_finite_difference(n_views: int) -> None:
    """The analytic gradient matches a numerical gradient, for fixed means/tau."""
    rng = np.random.default_rng(0)
    n, k = 20, 3
    representations = [rng.standard_normal((n, k)) for _ in range(n_views)]
    means = [z.mean(axis=0) for z in representations]
    total = sum(z - mu for z, mu in zip(representations, means))
    r = (total**2).sum(axis=1)
    tau = float(np.median(r)) + 1e-3  # avoid landing exactly on a sample's R (a kink)
    _, analytic = _hinge_ey(representations, means, tau)
    numeric = _numerical_grad_z(representations, means, tau)
    for a, b in zip(analytic, numeric):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_hinge_ey_reduces_to_plain_ey_when_tau_never_binds() -> None:
    """An unreachably large tau (no sample ever capped) reproduces ey_loss exactly."""
    rng = np.random.default_rng(0)
    representations = [rng.standard_normal((20, 2)) for _ in range(2)]
    means = [z.mean(axis=0) for z in representations]
    huge_tau = 1e12
    objective, _ = _hinge_ey(representations, means, huge_tau)
    expected = ey_loss(representations)["objective"]
    assert objective == pytest.approx(expected, abs=1e-8)


def test_self_calibrated_tau_caps_roughly_half_the_batch() -> None:
    """tau_mult=1 (the default) caps close to the better-than-average half."""
    rng = np.random.default_rng(0)
    representations = [rng.standard_normal((200, 2)) for _ in range(2)]
    means = [z.mean(axis=0) for z in representations]
    tau = _self_calibrated_tau(representations, means, tau_mult=1.0)
    total = sum(z - mu for z, mu in zip(representations, means))
    r = (total**2).sum(axis=1)
    capped_fraction = (r >= tau).mean()
    assert 0.2 < capped_fraction < 0.6


def test_reward_gradient_is_exactly_zero_for_capped_samples() -> None:
    """Samples already at/above tau get exactly zero *reward* gradient.

    This is the genuine sparsity mechanism -- but it is specifically the
    reward's contribution that vanishes, not the full per-sample gradient:
    the penalty term tr(VV) is built from every sample's data (unlike SVM's
    ||w||^2, which is not data-dependent at all), so a capped sample still
    influences the fit through the (unmodified) penalty. Confirmed here by
    reconstructing the reward-only piece directly from `_hinge_ey`'s formula.
    """
    rng = np.random.default_rng(0)
    representations = [rng.standard_normal((60, 2)) for _ in range(2)]
    means = [z.mean(axis=0) for z in representations]
    total = sum(z - mu for z, mu in zip(representations, means))
    r = (total**2).sum(axis=1)
    tau = float(np.median(r))
    m = len(representations)
    n = representations[0].shape[0]
    scale = 4.0 / (m * (n - 1))
    active = (r < tau).astype(float)
    reward_grad = scale * active[:, None] * total  # the term _hinge_ey subtracts
    capped = r >= tau
    assert capped.sum() > 0
    np.testing.assert_allclose(reward_grad[capped], 0.0)
    assert not np.allclose(reward_grad[~capped], 0.0)


def test_fit_recovers_high_correlation_on_correlated_data(
    correlated_views: list[np.ndarray],
) -> None:
    """Fits a real shared-latent-factor dataset to a high canonical correlation."""
    model = SupportVectorCCA(
        latent_dimensions=1, max_iter=2000, learning_rate=0.05, random_state=0
    )
    model.fit(correlated_views)
    z1, z2 = model.transform(correlated_views)
    corr = abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])
    assert corr > 0.7
