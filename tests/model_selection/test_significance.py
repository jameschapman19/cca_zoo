"""The permutation test of canonical correlations and loadings."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA
from cca_zoo.model_selection import permutation_test_significance


@pytest.fixture
def signal_and_noise() -> list[np.ndarray]:
    """Each view: shared-signal columns followed by three noise columns."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((80, 1))
    signal = [
        z @ rng.standard_normal((1, p)) + 0.2 * rng.standard_normal((80, p))
        for p in (4, 3)
    ]
    return [np.hstack([s, rng.standard_normal((80, 3))]) for s in signal]


def test_signal_features_are_more_significant_than_noise(
    signal_and_noise: list[np.ndarray],
) -> None:
    """Every signal column's loading is more significant than every noise column's."""
    result = permutation_test_significance(
        CCA(), signal_and_noise, n_permutations=199, random_state=0
    )
    p = result.loading_p_values[0][:, 0]
    assert p[:4].max() < p[4:].min()
    assert result.null_loadings[0].shape == (199, 7, 1)


def test_independent_views_are_not_significant() -> None:
    """Unrelated views give a large p-value for the correlation."""
    rng = np.random.default_rng(7)
    views = [rng.standard_normal((60, 5)) for _ in range(2)]
    result = permutation_test_significance(
        CCA(), views, n_permutations=99, random_state=0
    )
    assert result.p_values[0] > 0.1


def test_estimator_is_left_unfitted(signal_and_noise: list[np.ndarray]) -> None:
    """The test fits clones, as sklearn's permutation_test_score does."""
    estimator = CCA()
    permutation_test_significance(estimator, signal_and_noise, n_permutations=9)
    assert not hasattr(estimator, "weights_")
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_test_significance(estimator, signal_and_noise, n_permutations=0)
