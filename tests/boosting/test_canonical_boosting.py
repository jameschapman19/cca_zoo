"""Tests for CanonicalBoostingRegressor and CanonicalBoostingClassifier."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from cca_zoo.boosting import CanonicalBoostingClassifier, CanonicalBoostingRegressor


@parametrize_with_checks(
    [
        CanonicalBoostingRegressor(n_estimators=10, min_samples_leaf=1),
        CanonicalBoostingClassifier(n_estimators=10, min_samples_leaf=1),
    ]
)
def test_sklearn_estimator_checks(estimator: object, check: object) -> None:
    """Both estimators pass scikit-learn's full estimator contract."""
    check(estimator)


def _single_index(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10))
    return X, np.sin(2 * X @ np.ones(10) / np.sqrt(10))


def test_canonical_directions_beat_axis_aligned_on_oblique_signal() -> None:
    """On a single-index target, CCA features give far better generalisation."""
    X, y = _single_index(2000, 0)
    Xt, yt = _single_index(2000, 1)
    kw = dict(n_estimators=200, random_state=0)
    cano = CanonicalBoostingRegressor(**kw).fit(X, y).score(Xt, yt)
    axis = CanonicalBoostingRegressor(n_components=0, **kw).fit(X, y).score(Xt, yt)
    assert cano > 0.95
    assert cano - axis > 0.3


def test_first_direction_recovers_index_vector() -> None:
    """The first round's canonical direction aligns with the true index vector."""
    X, y = _single_index(2000, 0)
    model = CanonicalBoostingRegressor(n_estimators=1, random_state=0).fit(X, y)
    w = model.canonical_directions()[0, :, 0]
    assert abs(w @ np.ones(10) / np.sqrt(10)) > 0.95


def test_directions_limited_by_gradient_rank() -> None:
    """Directions per round are capped by the gradient matrix's rank."""
    X, y = _single_index(300, 0)
    reg = CanonicalBoostingRegressor(
        n_estimators=3, n_components=4, random_state=0
    ).fit(X, y)
    assert reg.canonical_directions().shape == (3, 10, 1)
    labels = np.digitize(y, [-0.5, 0.5])
    clf = CanonicalBoostingClassifier(
        n_estimators=3, n_components=4, random_state=0
    ).fit(X, labels)
    # softmax gradients sum to zero across classes, so rank is n_classes - 1
    assert clf.canonical_directions().shape == (3, 10, 2)


def test_zero_components_adds_no_features() -> None:
    """n_components=0 trains trees on the raw features only."""
    X, y = _single_index(300, 0)
    model = CanonicalBoostingRegressor(n_estimators=3, n_components=0).fit(X, y)
    assert all(r.tree.n_features_in_ == 10 for r in model.rounds_)


def test_multioutput_regression_shapes() -> None:
    """Multi-target regression predicts every output with one tree per round."""
    X, y = _single_index(300, 0)
    Y = np.column_stack([y, -y, y**2])
    model = CanonicalBoostingRegressor(n_estimators=20, random_state=0).fit(X, Y)
    assert model.predict(X).shape == (300, 3)
    assert len(model.rounds_) == 20


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_probabilities(n_classes: int) -> None:
    """predict_proba rows are valid distributions and predict is their argmax."""
    X, y = _single_index(400, 0)
    edges = np.linspace(-1, 1, n_classes + 1)[1:-1]
    labels = np.array(list("abc"))[np.digitize(y, edges)]
    model = CanonicalBoostingClassifier(n_estimators=30, random_state=0).fit(X, labels)
    P = model.predict_proba(X)
    assert P.shape == (400, n_classes)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.predict(X), model.classes_[P.argmax(1)])
    assert model.score(X, labels) > 0.9


def test_early_stopping_truncates_to_best_iteration() -> None:
    """Early stopping keeps exactly the rounds up to the best validation loss."""
    X, y = _single_index(400, 0)
    rng = np.random.default_rng(0)
    y_noisy = y + rng.standard_normal(400)
    model = CanonicalBoostingRegressor(
        n_estimators=500, learning_rate=0.3, random_state=0
    ).fit(
        X[:300],
        y_noisy[:300],
        eval_set=(X[300:], y_noisy[300:]),
        early_stopping_rounds=10,
    )
    best = int(np.argmin(model.eval_loss_)) + 1
    assert model.best_iteration_ == best == len(model.rounds_)
    assert len(model.eval_loss_) == best + 10
