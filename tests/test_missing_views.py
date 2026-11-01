"""The contract of the models that fit samples lacking some views.

A sample lacks a view when its whole row is NaN. Every model that declares
``_supports_missing_views`` is checked here, so a model that gains the flag
gets these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._base import BaseModel
from tests._helpers import (
    MODEL_CLASSES,
    fit,
    make_model,
    model_params,
    principal_cosines,
)

SUPPORTING = [c for c in MODEL_CLASSES if c._supports_missing_views]


def _incomplete_views(
    n: int = 300, noise: float = 0.5, seed: int = 0
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Three views of two shared factors; the complete views, the incomplete, factors.

    A quarter of the samples lack each view, in blocks, and a quarter have all.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    complete = [
        z @ rng.standard_normal((2, p)) + noise * rng.standard_normal((n, p))
        for p in (8, 6, 5)
    ]
    incomplete = [v.copy() for v in complete]
    q = n // 4
    for i in range(3):
        incomplete[i][i * q : (i + 1) * q] = np.nan
    return complete, incomplete, z


def test_the_models_that_support_missing_views() -> None:
    """The flag is on the models implemented, not on the robust ones."""
    names = {c.__name__ for c in SUPPORTING}
    assert {"IncompleteGCCA", "CCAEY", "PLSEY", "StochasticCCAEY"} <= names
    assert not names & {"HuberCCA", "TrimmedCCA", "GCCA", "CCA"}


@pytest.mark.parametrize("cls", model_params(SUPPORTING))
def test_reads_nan_as_a_missing_view(cls: type[BaseModel]) -> None:
    """The model advertises NaN input, which a model that does not rejects."""
    assert cls()._reads_nan
    assert cls().__sklearn_tags__().input_tags.allow_nan


@pytest.mark.parametrize("cls", model_params(SUPPORTING))
def test_recovers_the_shared_factors_with_views_missing(
    cls: type[BaseModel],
) -> None:
    """Every sample's shared score follows the factors, whichever views it has."""
    _, incomplete, z = _incomplete_views()
    model = make_model(cls).set_params(n_components=2)
    if "max_iter" in model.get_params():
        model.set_params(max_iter=500)
    fit(model, incomplete)
    shared = model.transform_shared(incomplete)
    assert shared.shape == (300, 2) and np.isfinite(shared).all()
    assert principal_cosines(shared, z).min() > 0.9
    assert model.score(incomplete) > 0.8


@pytest.mark.parametrize("cls", model_params(SUPPORTING))
def test_unobserved_samples_have_nan_scores_in_that_view_only(
    cls: type[BaseModel],
) -> None:
    """Transform keeps the NaN rows of a missing view and no others."""
    _, incomplete, _ = _incomplete_views()
    model = fit(make_model(cls).set_params(n_components=2), incomplete)
    for scores, view in zip(model.transform(incomplete), incomplete):
        missing = np.isnan(view).all(axis=1)
        assert np.isnan(scores[missing]).all()
        assert np.isfinite(scores[~missing]).all()
    predicted = model.predict([incomplete[0], None, None])
    assert all(
        np.isfinite(p[~np.isnan(incomplete[0]).all(axis=1)]).all() for p in predicted
    )


@pytest.mark.parametrize("cls", model_params(SUPPORTING))
def test_feature_means_and_importances_use_the_observed_samples(
    cls: type[BaseModel],
) -> None:
    """Centring and importances ignore the samples that lack the view."""
    _, incomplete, _ = _incomplete_views()
    shifted = [v + 10.0 for v in incomplete]
    model = fit(make_model(cls).set_params(n_components=2), shifted)
    for mean, view in zip(model.means_, shifted):
        np.testing.assert_allclose(mean, np.nanmean(view, axis=0))
    for importance in model.feature_importances_per_view_:
        assert np.isfinite(importance).all() and importance.sum() == pytest.approx(1)


@pytest.mark.parametrize("cls", model_params(SUPPORTING))
def test_rows_must_be_wholly_observed_or_missing(cls: type[BaseModel]) -> None:
    """A row with some entries missing raises, naming the rows."""
    _, incomplete, _ = _incomplete_views()
    incomplete[1][200, 2] = np.nan
    with pytest.raises(ValueError, match="rows \\[200\\] are partly missing"):
        fit(make_model(cls), incomplete)


@pytest.mark.parametrize("cls", model_params(SUPPORTING))
def test_every_sample_needs_a_view(cls: type[BaseModel]) -> None:
    """A sample missing from every view raises, naming the samples."""
    _, incomplete, _ = _incomplete_views()
    for v in incomplete:
        v[290] = np.nan
    with pytest.raises(ValueError, match="samples \\[290\\] have none"):
        fit(make_model(cls), incomplete)


@pytest.mark.parametrize("cls", model_params(SUPPORTING))
def test_unlinked_views_raise(cls: type[BaseModel]) -> None:
    """Views joined by too few shared samples have scores that are not tied together."""
    views, _, _ = _incomplete_views(n=60)
    # Views 0 and 1 have samples 0-29; view 2 has 28-59, so shares only 2.
    for view, rows in zip(views, [slice(30, 60), slice(30, 60), slice(0, 28)]):
        view[rows] = np.nan
    incomplete = views
    with pytest.raises(ValueError, match="not linked"):
        fit(make_model(cls).set_params(n_components=2), incomplete)
