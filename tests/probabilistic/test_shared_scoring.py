"""Regression tests for scoring/loadings shared by every probabilistic model.

``ProbabilisticCCA`` and ``VariationalBayesCCA`` both return a *single*
shared-latent array from ``transform`` (there is one joint z, not one
per-view canonical variate). ``BaseModel``'s default ``score`` /
``pairwise_correlations`` / ``get_factor_loadings`` assume one array per
view and silently misbehave when only one is returned:
``average_pairwise_correlations`` degenerates a 2-view problem to a 1x1
self-comparison (0/0 -> nan), and ``get_factor_loadings`` zips the single
array against every view, silently truncating to just the first one. Both
are fixed by ``cca_zoo.probabilistic._utils.PosteriorMeanTransformMixin``.
All tests are marked slow and require numpyro + jax.
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
        return dict(num_warmup=20, num_samples=20)
    return dict(num_steps=300)


def _model_classes() -> list[type]:
    names = ["ProbabilisticCCA", "VariationalBayesCCA"]
    return [getattr(pcca_module, n) for n in names if hasattr(pcca_module, n)]


@pytest.fixture
def two_views() -> list[np.ndarray]:
    """Two small random views."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal((30, 4)), rng.standard_normal((30, 3))]


@pytest.fixture
def three_views() -> list[np.ndarray]:
    """Three small random views."""
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal((30, 4)),
        rng.standard_normal((30, 3)),
        rng.standard_normal((30, 5)),
    ]


@pytest.mark.slow
@pytest.mark.parametrize(
    "ModelClass", _model_classes(), ids=[c.__name__ for c in _model_classes()]
)
def test_score_is_finite_two_views(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """score() must not be nan for a 2-view fit (regression for the 0/0 bug)."""
    model = ModelClass(
        latent_dimensions=2, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (2,)
    assert np.all(np.isfinite(s))


@pytest.mark.slow
@pytest.mark.parametrize(
    "ModelClass", _model_classes(), ids=[c.__name__ for c in _model_classes()]
)
def test_score_is_finite_three_views(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """score() must not be nan for a 3-view fit."""
    model = ModelClass(
        latent_dimensions=1, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(three_views)
    s = model.score(three_views)
    assert s.shape == (1,)
    assert np.all(np.isfinite(s))


@pytest.mark.slow
@pytest.mark.parametrize(
    "ModelClass", _model_classes(), ids=[c.__name__ for c in _model_classes()]
)
def test_pairwise_correlations_shape(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """pairwise_correlations returns (n_views, n_views, k), not degenerate (1,1,k)."""
    model = ModelClass(
        latent_dimensions=2, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(three_views)
    corrs = model.pairwise_correlations(three_views)
    assert corrs.shape == (3, 3, 2)
    assert np.all(np.isfinite(corrs))


@pytest.mark.slow
@pytest.mark.parametrize(
    "ModelClass", _model_classes(), ids=[c.__name__ for c in _model_classes()]
)
def test_get_factor_loadings_one_per_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """get_factor_loadings returns one array per view, not just the first."""
    k = 2
    model = ModelClass(
        latent_dimensions=k, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(three_views)
    loadings = model.get_factor_loadings(three_views)
    assert len(loadings) == 3
    for loading, view in zip(loadings, three_views):
        assert loading.shape == (view.shape[1], k)


@pytest.mark.slow
@pytest.mark.parametrize(
    "ModelClass", _model_classes(), ids=[c.__name__ for c in _model_classes()]
)
def test_log_likelihood_is_finite_scalar(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """log_likelihood returns a finite scalar, evaluated jointly across views."""
    model = ModelClass(
        latent_dimensions=2, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(three_views)
    ll = model.log_likelihood(three_views)
    assert isinstance(ll, float)
    assert np.isfinite(ll)


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
        latent_dimensions=k, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(good_views)
    bad_model = ModelClass(
        latent_dimensions=k, random_state=0, **_fast_kwargs(ModelClass)
    ).fit(bad_views)

    assert good_model.log_likelihood(good_views) > bad_model.log_likelihood(good_views)
