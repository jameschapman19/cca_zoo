"""ProbabilisticCCA (NUTS) and VariationalBayesCCA (SVI), which need numpyro."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro")

from cca_zoo.linear import CCA
from cca_zoo.probabilistic import ProbabilisticCCA, VariationalBayesCCA

pytestmark = pytest.mark.slow

_QUICK = {
    ProbabilisticCCA: {"n_warmup": 20, "n_posterior_samples": 20},
    VariationalBayesCCA: {"n_iter": 300},
}


def _views(
    n: int = 150, k: int = 2, noise: np.ndarray | float = 0.1
) -> tuple[list[np.ndarray], np.ndarray]:
    """Two views of ``k`` shared factors, and the factors."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, k))
    sd = [np.broadcast_to(noise, 6), np.full(5, 0.1)]
    views = [
        z @ rng.standard_normal((k, len(s))) + rng.standard_normal((n, len(s))) * s
        for s in sd
    ]
    return views, z


@pytest.mark.parametrize("cls", [ProbabilisticCCA, VariationalBayesCCA])
def test_transform_is_each_views_posterior_mean(cls: type) -> None:
    """Each view's scores are the latent's posterior mean given that view alone."""
    views, _ = _views(n=40)
    model = cls(n_components=2, **_QUICK[cls]).fit(views)
    for i, scores in enumerate(model.transform(views)):
        observed = [v if j == i else None for j, v in enumerate(views)]
        np.testing.assert_allclose(scores, model.posterior_mean(observed), atol=1e-6)


@pytest.mark.parametrize("cls", [ProbabilisticCCA, VariationalBayesCCA])
def test_log_likelihood_prefers_the_model_fitted_to_the_data(cls: type) -> None:
    """A model fitted to the views explains them better than one fitted to noise."""
    views, _ = _views(n=200, k=1)
    noise = [np.random.default_rng(1).standard_normal(v.shape) for v in views]
    fitted = cls(random_state=0, **_QUICK[cls]).fit(views)
    unrelated = cls(random_state=0, **_QUICK[cls]).fit(noise)
    assert fitted.log_likelihood(views) > unrelated.log_likelihood(views)


def test_noise_is_a_variance() -> None:
    """The fitted noise matches each feature's true variance, 0.01 or 4."""
    views, _ = _views(n=400, noise=np.array([0.1] * 3 + [2.0] * 3))
    psi = (
        VariationalBayesCCA(2, n_iter=3000, random_state=0)
        .fit(views)
        ._noise_variances()[0]
    )
    assert psi[:3].max() < 0.05 and psi[3:].min() > 3.0


def test_vb_recovers_the_latent_subspace() -> None:
    """The posterior mean spans the true latent space, up to rotation."""
    views, z = _views()
    z_hat = (
        VariationalBayesCCA(2, n_iter=2000, random_state=0)
        .fit(views)
        .posterior_mean(views)
    )
    assert CCA(2).fit([z, z_hat]).score([z, z_hat]) > 0.8


def test_ard_shrinks_an_unsupported_dimension() -> None:
    """Of three dimensions for two factors, the spare one gets the largest precision."""
    views, _ = _views()
    relevance = np.sort(
        VariationalBayesCCA(3, n_iter=2000, random_state=0).fit(views).ard_relevance_
    )
    assert relevance[2] > 2 * relevance[1]


def test_nuts_draws_share_one_rotation() -> None:
    """Aligned draws agree, so averaging them does not shrink the loadings."""
    views, _ = _views(n=100)
    model = ProbabilisticCCA(
        2, n_warmup=500, n_posterior_samples=1000, random_state=0
    ).fit(views)
    draws = np.concatenate(
        [model.posterior_samples_[f"W_{i}"] for i in range(2)], axis=1
    )
    coherence = np.linalg.norm(draws.mean(axis=0)) ** 2 / np.mean(
        np.linalg.norm(draws, axis=(1, 2)) ** 2
    )
    assert coherence > 0.95
