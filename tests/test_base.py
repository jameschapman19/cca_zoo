"""BaseModel's shared score, predict and inverse_transform, on CCA."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA
from cca_zoo.metrics import average_pairwise_correlations, pairwise_correlations


@pytest.fixture
def scaled_views(correlated_views: list[np.ndarray]) -> list[np.ndarray]:
    """Correlated views with features on very different scales."""
    rng = np.random.default_rng(1)
    return [v * rng.uniform(0.5, 20, size=v.shape[1]) for v in correlated_views]


def test_score_is_the_mean_canonical_correlation(
    two_views: list[np.ndarray],
) -> None:
    """Score is the mean over dimensions of the per-dimension correlations."""
    model = CCA(n_components=2).fit(two_views)
    per_dimension = average_pairwise_correlations(
        pairwise_correlations(model.transform(two_views))
    )
    assert model.score(two_views) == pytest.approx(per_dimension.mean(), rel=1e-12)


def test_predict_reconstructs_another_view(scaled_views: list[np.ndarray]) -> None:
    """Predict's least-squares loadings beat the naive ``scores @ weights.T``.

    Weights are not loadings unless the data is whitened (#182).
    """
    model = CCA(n_components=2).fit(scaled_views)
    predicted = model.predict([scaled_views[0], None])[1]
    scores = (scaled_views[0] - model.means_[0]) @ model.weights_[0]
    naive = scores @ model.weights_[1].T + model.means_[1]

    def fidelity(x: np.ndarray) -> float:
        return float(np.corrcoef(x.ravel(), scaled_views[1].ravel())[0, 1])

    assert fidelity(predicted) > 0.9
    assert fidelity(predicted) > fidelity(naive)


def test_predict_needs_an_observed_view(two_views: list[np.ndarray]) -> None:
    """Predict raises when every view is None."""
    with pytest.raises(ValueError, match="At least one view"):
        CCA().fit(two_views).predict([None, None])


def test_inverse_transform_round_trips(scaled_views: list[np.ndarray]) -> None:
    """inverse_transform(transform(views)) recovers each view from its own scores."""
    model = CCA(n_components=2).fit(scaled_views)
    for approx, view in zip(
        model.inverse_transform(model.transform(scaled_views)), scaled_views
    ):
        assert np.corrcoef(approx.ravel(), view.ravel())[0, 1] > 0.9


def test_inverse_transform_is_not_predict(correlated_views: list[np.ndarray]) -> None:
    """inverse_transform uses a view's own scores; predict uses the others'."""
    model = CCA(n_components=2).fit(correlated_views)
    own = model.inverse_transform(model.transform(correlated_views))[1]
    assert not np.allclose(own, model.predict([correlated_views[0], None])[1])


def test_inverse_transform_checks_the_scores(two_views: list[np.ndarray]) -> None:
    """One score array per view, each n_components wide."""
    model = CCA(n_components=2).fit(two_views)
    scores = model.transform(two_views)
    with pytest.raises(ValueError, match="Expected 2 score arrays"):
        model.inverse_transform(scores[:1])
    with pytest.raises(ValueError, match="expected 2"):
        model.inverse_transform([scores[0][:, :1], scores[1]])
