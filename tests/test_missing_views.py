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


def _wide_views(
    shared: float, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Wide views of two shared and three private factors, with views missing.

    Returns the NaN-masked views, each view's noise-free shared signal, and the
    boolean array of which samples have which views.
    """
    rng = np.random.default_rng(seed)
    n, widths = 150, (150, 200, 100)
    z = rng.standard_normal((n, 2))
    signals = [z @ rng.standard_normal((2, p)) * np.sqrt(shared / 2) for p in widths]
    views = [
        s
        + rng.standard_normal((n, 3)) @ rng.standard_normal((3, p)) * np.sqrt(0.8 / 3)
        + rng.standard_normal((n, p)) * np.sqrt(0.2 - shared)
        for s, p in zip(signals, widths)
    ]
    observed = rng.random((n, 3)) >= 0.3
    for i in np.flatnonzero(~observed.any(axis=1)):
        observed[i, rng.integers(3)] = True
    masked = [np.where(o[:, None], v, np.nan) for v, o in zip(views, observed.T)]
    return masked, signals, observed


def _fill_in_error(model: BaseModel, shared: float) -> float:
    """Error of filling in the withheld views, over that of predicting their mean."""
    ratios = []
    for seed in range(3):
        masked, signals, observed = _wide_views(shared, seed)
        fit(model, masked)
        error = baseline = 0.0
        for pattern in np.unique(observed, axis=0):
            rows = np.flatnonzero((observed == pattern).all(axis=1))
            predicted = model.predict(
                [m[rows] if p else None for m, p in zip(masked, pattern)]
            )
            for j in np.flatnonzero(~pattern):
                error += np.sum((predicted[j] - signals[j][rows]) ** 2)
                baseline += np.sum(
                    (np.nanmean(masked[j], axis=0) - signals[j][rows]) ** 2
                )
        ratios.append(error / baseline)
    return float(np.mean(ratios))


# StochasticCCAEY's quick test settings are too short to converge.
_FILLERS = [c for c in SUPPORTING if c.__name__ != "StochasticCCAEY"]


@pytest.mark.parametrize("cls", model_params(_FILLERS))
def test_filling_in_a_view_is_calibrated_to_how_much_the_others_tell(
    cls: type[BaseModel],
) -> None:
    """A faint shared signal gives no worse than the mean; a strong one beats it.

    The scores of a model fitted to wide views overfit the target view, so a map
    fitted to them predicts far beyond what the other views know: about seven times
    the error of predicting the mean when the shared signal is 1% of the variance.
    """
    model = make_model(cls).set_params(n_components=2)
    if "shrinkage" in model.get_params():
        model.set_params(shrinkage=0.2)
    assert _fill_in_error(model, shared=0.01) < 2.0
    # PLS maximises covariance, so private factors with much of the variance pull
    # its directions off the shared ones; it still does better than the mean.
    strong = 1.0 if cls.__name__ == "PLSEY" else 0.6
    assert _fill_in_error(model, shared=0.1) < strong


def test_predict_maps_fall_back_with_a_warning_when_the_folds_cannot_be_fitted() -> (
    None
):
    """With too few samples shared by the views to cross-fit, the model says so."""
    from cca_zoo.linear import IncompleteGCCA

    views, _, _ = _incomplete_views(n=60)
    for view, rows in zip(views, [slice(30, 60), slice(30, 60), slice(0, 27)]):
        view[rows] = np.nan
    with pytest.warns(UserWarning, match="could not cross-fit its predict maps"):
        IncompleteGCCA(n_components=2).fit(views)


@pytest.mark.parametrize("cls", model_params(_FILLERS))
def test_impute_fills_the_missing_views_and_keeps_the_observed(
    cls: type[BaseModel],
) -> None:
    """Missing rows are predicted, observed rows are returned unchanged."""
    _, incomplete, _ = _incomplete_views()
    model = fit(make_model(cls).set_params(n_components=2), incomplete)
    filled = model.impute(incomplete)
    for original, completed in zip(incomplete, filled):
        missing = np.isnan(original).all(axis=1)
        assert np.isfinite(completed).all()
        np.testing.assert_array_equal(completed[~missing], original[~missing])
        assert not np.allclose(completed[missing], 0)


def test_impute_needs_a_model_that_fits_missing_views() -> None:
    """A model that requires every view has nothing to impute."""
    from cca_zoo.linear import CCA

    views, _, _ = _incomplete_views()
    model = CCA().fit([v[100:200] for v in views[:2]])
    with pytest.raises(ValueError, match="does not fit samples that lack views"):
        model.impute(views[:2])


def test_impute_rescales_the_scores_of_the_samples_it_was_fitted_to() -> None:
    """With views of thousands of features, fitted samples' scores are inflated.

    The weights were fitted to those samples, so their scores spread far wider
    than the out-of-fold scores the prediction map was fitted to. Rescaling them
    does better than predicting from them as they are.
    """
    from cca_zoo.linear import IncompleteGCCA

    for seed in range(3):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((80, 2))
        signal = [
            z @ rng.standard_normal((2, p)) * np.sqrt(0.025) for p in (2000, 1500, 1000)
        ]
        views = [s + rng.standard_normal(s.shape) * np.sqrt(0.95) for s in signal]
        views[0][:20] = np.nan
        model = IncompleteGCCA(n_components=2, shrinkage=0.5).fit(views)
        mean = np.nanmean(views[0], axis=0)

        def error(filled: list[np.ndarray]) -> float:
            return float(
                np.sum((filled[0][:20] - signal[0][:20]) ** 2)
                / np.sum((mean - signal[0][:20]) ** 2)
            )

        rescaled = error(model.impute(views))
        assert rescaled < error(model.impute(views, fitted=False))
        assert rescaled < 0.8  # and it still beats predicting the view's mean
