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


def test_signal_features_are_significant_and_noise_is_not(
    signal_and_noise: list[np.ndarray],
) -> None:
    """Loadings on the signal columns have small p-values; noise columns do not."""
    result = permutation_test_significance(
        CCA(), signal_and_noise, n_permutations=199, random_state=0
    )
    p = result.loading_p_values[0][:, 0]
    assert np.all(p[:4] < 0.1) and np.all(p[4:] > 0.1)
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
