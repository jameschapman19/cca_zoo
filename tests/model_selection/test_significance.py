"""Tests for cca_zoo.model_selection's permutation testing."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA
from cca_zoo.model_selection import (
    PermutationTestResult,
    permutation_test_significance,
    procrustes_rotation,
)

# ---------------------------------------------------------------------------
# procrustes_rotation
# ---------------------------------------------------------------------------


def test_procrustes_rotation_is_deprecated_scipy() -> None:
    """procrustes_rotation warns and matches scipy.linalg.orthogonal_procrustes."""
    import scipy.linalg

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((20, 3))
    target = reference @ np.linalg.qr(rng.standard_normal((3, 3)))[0].T
    with pytest.warns(FutureWarning, match="orthogonal_procrustes"):
        rotation = procrustes_rotation(reference, target)
    np.testing.assert_allclose(
        rotation, scipy.linalg.orthogonal_procrustes(target, reference)[0]
    )
    np.testing.assert_allclose(target @ rotation, reference, atol=1e-10)


# ---------------------------------------------------------------------------
# permutation_test_significance
# ---------------------------------------------------------------------------


@pytest.fixture
def signal_and_noise_views() -> list[np.ndarray]:
    """Two views each with a shared-signal block and a pure-noise block."""
    rng = np.random.default_rng(0)
    n = 80
    z = rng.standard_normal((n, 1))
    signal1 = z @ rng.standard_normal((1, 4)) + 0.2 * rng.standard_normal((n, 4))
    signal2 = z @ rng.standard_normal((1, 3)) + 0.2 * rng.standard_normal((n, 3))
    noise1 = rng.standard_normal((n, 3))
    noise2 = rng.standard_normal((n, 3))
    return [np.hstack([signal1, noise1]), np.hstack([signal2, noise2])]


def test_permutation_test_returns_result_object(
    signal_and_noise_views: list[np.ndarray],
) -> None:
    """permutation_test_significance returns a PermutationTestResult."""
    result = permutation_test_significance(
        CCA(latent_dimensions=1),
        signal_and_noise_views,
        n_permutations=19,
        random_state=0,
    )
    assert isinstance(result, PermutationTestResult)


def test_permutation_test_shapes(signal_and_noise_views: list[np.ndarray]) -> None:
    """All result arrays have the expected shapes."""
    k = 1
    n_perm = 19
    result = permutation_test_significance(
        CCA(latent_dimensions=k),
        signal_and_noise_views,
        n_permutations=n_perm,
        random_state=0,
    )
    assert result.correlations_.shape == (k,)
    assert result.null_correlations_.shape == (n_perm, k)
    assert result.p_values_.shape == (k,)
    assert len(result.loadings_) == 2
    assert len(result.null_loadings_) == 2
    assert len(result.loading_p_values_) == 2
    for view, loading, null_loading, loading_p in zip(
        signal_and_noise_views,
        result.loadings_,
        result.null_loadings_,
        result.loading_p_values_,
    ):
        assert loading.shape == (view.shape[1], k)
        assert null_loading.shape == (n_perm, view.shape[1], k)
        assert loading_p.shape == (view.shape[1], k)


def test_permutation_test_p_values_in_valid_range(
    signal_and_noise_views: list[np.ndarray],
) -> None:
    """p_values_ and loading_p_values_ all lie in (0, 1]."""
    result = permutation_test_significance(
        CCA(latent_dimensions=1),
        signal_and_noise_views,
        n_permutations=19,
        random_state=0,
    )
    assert np.all(result.p_values_ > 0) and np.all(result.p_values_ <= 1)
    for p in result.loading_p_values_:
        assert np.all(p > 0) and np.all(p <= 1)


def test_permutation_test_signal_features_more_significant_than_noise(
    signal_and_noise_views: list[np.ndarray],
) -> None:
    """Signal-block loadings get lower p-values than pure-noise-block loadings.

    This is the core ask of #130: telling apart features that reliably
    drive a canonical dimension from features that don't.
    """
    result = permutation_test_significance(
        CCA(latent_dimensions=1),
        signal_and_noise_views,
        n_permutations=199,
        random_state=0,
    )
    # View 0: first 4 columns are signal, last 3 are noise.
    signal_p = result.loading_p_values_[0][:4, 0]
    noise_p = result.loading_p_values_[0][4:, 0]
    assert signal_p.mean() < noise_p.mean()
    assert np.all(signal_p < 0.1)
    assert np.all(noise_p > 0.1)


def test_permutation_test_unrelated_views_not_significant() -> None:
    """Independent views give a large (non-significant) correlation p-value."""
    rng = np.random.default_rng(7)
    x1 = rng.standard_normal((60, 5))
    x2 = rng.standard_normal((60, 5))
    result = permutation_test_significance(
        CCA(latent_dimensions=1), [x1, x2], n_permutations=99, random_state=0
    )
    assert result.p_values_[0] > 0.1


def test_permutation_test_reproducible_with_same_random_state(
    signal_and_noise_views: list[np.ndarray],
) -> None:
    """Same random_state gives identical results."""
    result_a = permutation_test_significance(
        CCA(latent_dimensions=1),
        signal_and_noise_views,
        n_permutations=15,
        random_state=42,
    )
    result_b = permutation_test_significance(
        CCA(latent_dimensions=1),
        signal_and_noise_views,
        n_permutations=15,
        random_state=42,
    )
    np.testing.assert_array_equal(
        result_a.null_correlations_, result_b.null_correlations_
    )


def test_permutation_test_does_not_mutate_estimator(
    signal_and_noise_views: list[np.ndarray],
) -> None:
    """The passed-in estimator is cloned, not fitted in place."""
    estimator = CCA(latent_dimensions=1)
    permutation_test_significance(
        estimator, signal_and_noise_views, n_permutations=9, random_state=0
    )
    assert not hasattr(estimator, "weights_")


def test_permutation_test_n_jobs(signal_and_noise_views: list[np.ndarray]) -> None:
    """permutation_test_significance works with n_jobs=2."""
    result = permutation_test_significance(
        CCA(latent_dimensions=1),
        signal_and_noise_views,
        n_permutations=9,
        random_state=0,
        n_jobs=2,
    )
    assert result.correlations_.shape == (1,)


def test_permutation_test_invalid_n_permutations_raises(
    signal_and_noise_views: list[np.ndarray],
) -> None:
    """n_permutations must be positive."""
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_test_significance(
            CCA(latent_dimensions=1), signal_and_noise_views, n_permutations=0
        )
