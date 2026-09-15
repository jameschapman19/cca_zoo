"""Tests for cca_zoo.linear._ransac_cca (robust multiview CCA via RANSAC)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import MCCA, RANSACCCA, HuberCCA
from cca_zoo.linear._ransac_cca import _cross_view_agreement


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> RANSACCCA:
    return RANSACCCA(
        latent_dimensions=latent_dimensions, max_trials=30, random_state=0, **kwargs
    )


# ---------------------------------------------------------------------------
# _cross_view_agreement
# ---------------------------------------------------------------------------


def test_agreement_positive_for_correlated_views() -> None:
    """Perfectly correlated views score positive agreement everywhere."""
    rng = np.random.default_rng(0)
    t = rng.standard_normal((50, 1))
    agreement = _cross_view_agreement([t, t])
    assert np.all(agreement > 0)


def test_agreement_negative_for_anticorrelated_views() -> None:
    """Perfectly anti-correlated views score negative agreement everywhere."""
    rng = np.random.default_rng(0)
    t = rng.standard_normal((50, 1))
    agreement = _cross_view_agreement([t, -t])
    assert np.all(agreement < 0)


def test_agreement_symmetric_in_view_order() -> None:
    """Agreement doesn't depend on the order views are passed in."""
    rng = np.random.default_rng(0)
    z1 = rng.standard_normal((30, 2))
    z2 = rng.standard_normal((30, 2))
    z3 = rng.standard_normal((30, 2))
    np.testing.assert_allclose(
        _cross_view_agreement([z1, z2, z3]), _cross_view_agreement([z3, z1, z2])
    )


# ---------------------------------------------------------------------------
# fit completes / shapes
# ---------------------------------------------------------------------------


def test_two_view_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = _make_model()
    fitted = model.fit(two_views_small)
    assert fitted is model


def test_three_view_fit_completes(three_views_small: list[np.ndarray]) -> None:
    """Fit completes on three-view data without error."""
    model = _make_model()
    fitted = model.fit(three_views_small)
    assert fitted is model


def test_weights_shapes_and_matches_transform(
    two_views_small: list[np.ndarray],
) -> None:
    """Weights are real (p_i, k) arrays and transform(v) == centred(v) @ weights."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    weights = model.weights
    assert len(weights) == 2
    for w, v in zip(weights, two_views_small):
        assert w.shape == (v.shape[1], k)

    transformed = model.transform(two_views_small)
    for v, w, t, mean in zip(two_views_small, weights, transformed, model.means_):
        np.testing.assert_allclose((v - mean) @ w, t, atol=1e-8)


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = RANSACCCA()
    with pytest.raises(NotFittedError):
        _ = model.weights


def test_inlier_mask_shape(two_views_small: list[np.ndarray]) -> None:
    """inlier_mask_ is a boolean mask over the training samples."""
    model = _make_model().fit(two_views_small)
    assert model.inlier_mask_.shape == (two_views_small[0].shape[0],)
    assert model.inlier_mask_.dtype == bool


def test_min_samples_as_int(two_views_small: list[np.ndarray]) -> None:
    """An absolute int min_samples is honoured directly."""
    n = two_views_small[0].shape[0]
    model = _make_model(min_samples=max(4, n // 2)).fit(two_views_small)
    assert model.inlier_mask_.shape == (n,)


# ---------------------------------------------------------------------------
# Correctness on clean data
# ---------------------------------------------------------------------------


def test_ransac_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """On clean, uncontaminated data RANSACCCA still finds real correlation."""
    model = RANSACCCA(latent_dimensions=1, max_trials=100, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


# ---------------------------------------------------------------------------
# The headline scenario: structured, leverage-invisible contamination
# ---------------------------------------------------------------------------


def _make_sign_flip_contaminated_data(
    seed: int, n_train: int, n_test: int, p1: int, p2: int, contam_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two views sharing one real latent factor, a fraction sign-flipped.

    Every row -- contaminated or not -- is drawn from the *same* generative
    process and the *same* noise scale, so a contaminated row is not a
    high-leverage outlier in either view on its own (unlike
    ``test_huber_cca.py``'s contamination, a deliberately large-magnitude
    cluster). What's wrong with it is purely relational: for a
    contaminated row, view 2's contribution is generated from the
    *negative* of the same latent factor that drives view 1, so the pair
    is anti-correlated exactly where the rest of the data is correlated.
    Nothing about its norm gives it away.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(p1)
    w1 /= np.linalg.norm(w1)
    w2 = rng.standard_normal(p2)
    w2 /= np.linalg.norm(w2)

    def group(n: int, sign: float) -> tuple[np.ndarray, np.ndarray]:
        t = rng.standard_normal(n)
        X = np.outer(t, w1) + 0.6 * rng.standard_normal((n, p1))
        Y = np.outer(sign * t, w2) + 0.6 * rng.standard_normal((n, p2))
        return X, Y

    x_test, y_test = group(n_test, sign=1.0)

    n_bad = int(round(contam_frac * n_train))
    n_good = n_train - n_bad
    x_good, y_good = group(n_good, sign=1.0)
    x_bad, y_bad = group(n_bad, sign=-1.0)
    x_train = np.vstack([x_good, x_bad])
    y_train = np.vstack([y_good, y_bad])
    contaminated = np.zeros(n_train, dtype=bool)
    contaminated[n_good:] = True

    perm = rng.permutation(n_train)
    return (
        x_train[perm],
        y_train[perm],
        x_test,
        y_test,
        contaminated[perm],
        np.array([n_good, n_bad]),
    )


def _held_out_corr(
    model: MCCA | HuberCCA | RANSACCCA, x_test: np.ndarray, y_test: np.ndarray
) -> float:
    z1, z2 = model.transform([x_test, y_test])
    return abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])


def test_ransac_robust_to_sign_flipped_rows_unlike_mcca_and_huber() -> None:
    """40% sign-flipped rows wreck plain MCCA and HuberCCA but not this.

    HuberCCA guards against high-*leverage* contamination; a sign-flipped
    row is completely ordinary in magnitude in both views, so Huber's
    leverage-based reweighting has nothing to grab onto and doesn't help
    (see ``_make_sign_flip_contaminated_data``'s docstring) -- it can even
    do slightly worse than plain MCCA, since it downweights some
    genuinely informative high-variance points without touching any of
    the actually-bad ones. RANSACCCA's random-subset consensus targets
    the right thing instead: whether a row's cross-view relationship
    agrees with the majority's, not how large it is.
    """
    x_train, y_train, x_test, y_test, contaminated, _ = (
        _make_sign_flip_contaminated_data(
            seed=0, n_train=400, n_test=300, p1=8, p2=6, contam_frac=0.4
        )
    )

    mcca_corr = _held_out_corr(
        MCCA(latent_dimensions=1, c=0.1).fit([x_train, y_train]), x_test, y_test
    )
    huber_corr = _held_out_corr(
        HuberCCA(latent_dimensions=1, max_iter=1500, random_state=0).fit(
            [x_train, y_train]
        ),
        x_test,
        y_test,
    )
    ransac_model = RANSACCCA(latent_dimensions=1, random_state=0).fit(
        [x_train, y_train]
    )
    ransac_corr = _held_out_corr(ransac_model, x_test, y_test)

    assert ransac_corr > mcca_corr + 0.3
    assert ransac_corr > huber_corr + 0.3

    # The consensus set found should be predominantly the real, uncontaminated
    # rows -- not a majority-vote guarantee in general, but true here.
    inliers = ransac_model.inlier_mask_
    assert (~contaminated[inliers]).mean() > 0.7


# ---------------------------------------------------------------------------
# sklearn compatibility spot-checks
# ---------------------------------------------------------------------------


def test_clone_and_get_params_roundtrip() -> None:
    """clone()/get_params() round-trip correctly (sklearn BaseEstimator contract)."""
    from sklearn.base import clone

    model = RANSACCCA(
        latent_dimensions=2, c=0.2, min_samples=0.3, max_trials=50, random_state=0
    )
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_invalid_max_trials_raises() -> None:
    """max_trials below 1 is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        RANSACCCA(max_trials=0)._validate_params()


def test_invalid_stop_probability_raises() -> None:
    """stop_probability outside [0, 1] is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        RANSACCCA(stop_probability=1.5)._validate_params()
