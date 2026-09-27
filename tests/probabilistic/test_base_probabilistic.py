"""The posterior inference shared by ProbabilisticCCA and VariationalBayesCCA.

All tests are marked slow and require numpyro and jax.
"""

from __future__ import annotations

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro", reason="numpyro is not installed")
jax = pytest.importorskip("jax", reason="jax is not installed")

pcca_module = pytest.importorskip(
    "cca_zoo.probabilistic",
    reason="cca_zoo.probabilistic could not be imported",
)


def _fast_kwargs(cls: type) -> dict:
    """Return kwargs that make ``cls`` fit quickly for a test."""
    if cls.__name__ == "ProbabilisticCCA":
        return dict(n_warmup=20, n_posterior_samples=20)
    return dict(max_iter=300)


def _model_classes() -> list[type]:
    names = ["ProbabilisticCCA", "VariationalBayesCCA"]
    return [getattr(pcca_module, n) for n in names if hasattr(pcca_module, n)]


@pytest.fixture
def two_views() -> list[np.ndarray]:
    """Two small random views."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal((30, 4)), rng.standard_normal((30, 3))]


@pytest.mark.slow
@pytest.mark.parametrize(
    "ModelClass", _model_classes(), ids=[c.__name__ for c in _model_classes()]
)
def test_log_likelihood_prefers_better_fit(ModelClass: type) -> None:
    """A model fit to correlated views scores better than one fit to noise.

    Compares log-likelihood on the same held-in correlated data between a
    model actually fit to it and a model fit to unrelated, uncorrelated
    views, sanity-checking that log_likelihood responds to fit quality
    rather than being a constant or a shape-only computation.
    """
    rng = np.random.default_rng(0)
    n, k = 200, 1
    z = rng.standard_normal((n, k))
    x1 = z @ rng.standard_normal((k, 4)) + 0.05 * rng.standard_normal((n, 4))
    x2 = z @ rng.standard_normal((k, 3)) + 0.05 * rng.standard_normal((n, 3))
    good_views = [x1, x2]

    bad_views = [rng.standard_normal((n, 4)), rng.standard_normal((n, 3))]

    good_model = ModelClass(
        n_components=k, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(good_views)
    bad_model = ModelClass(
        n_components=k, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(bad_views)

    assert good_model.log_likelihood(good_views) > bad_model.log_likelihood(good_views)


def _heteroscedastic_views(n: int = 400) -> tuple[list[np.ndarray], np.ndarray]:
    """Two views of a 2-d latent; view 1 has three quiet and three noisy features."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 2))
    sd = [np.array([0.1] * 3 + [2.0] * 3), np.full(5, 0.5)]
    views = [
        z @ rng.standard_normal((len(s), 2)).T + rng.standard_normal((n, len(s))) * s
        for s in sd
    ]
    return views, z


@pytest.mark.slow
def test_noise_parameter_is_a_variance() -> None:
    """The fitted noise matches the true variances (0.01 and 4), not their roots."""
    views, _ = _heteroscedastic_views()
    model = pcca_module.VariationalBayesCCA(2, max_iter=3000, random_state=0).fit(views)
    psi = model._noise_variances()[0]
    assert psi[:3].max() < 0.05
    assert psi[3:].min() > 3.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "ModelClass", _model_classes(), ids=[c.__name__ for c in _model_classes()]
)
def test_transform_is_each_views_posterior_mean(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """Each view's transform is its posterior mean given that view alone."""
    model = ModelClass(n_components=2, **_fast_kwargs(ModelClass)).fit(two_views)
    x1, x2 = model.transform(two_views)
    np.testing.assert_allclose(
        x1, model.posterior_mean([two_views[0], None]), atol=1e-6
    )
    np.testing.assert_allclose(
        x2, model.posterior_mean([None, two_views[1]]), atol=1e-6
    )
