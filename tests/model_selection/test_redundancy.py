"""The check for coordinates that are functions of earlier ones."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from cca_zoo.model_selection import nonlinear_redundancy


def _factors() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((1500, 3))


def test_independent_coordinates_are_not_redundant() -> None:
    """Nothing predicts an independent coordinate; the first has nothing before it."""
    redundancy = nonlinear_redundancy(_factors(), random_state=0)
    assert redundancy.shape == (3,)
    assert redundancy[0] == 0.0
    assert redundancy.max() < 0.05


def test_a_function_of_an_earlier_coordinate_is_flagged_though_uncorrelated() -> None:
    """z^2 - 1 is uncorrelated with z, which EY cannot see, but a function of it."""
    z = _factors()
    embedding = np.column_stack([z[:, 0], z[:, 0] ** 2 - 1, z[:, 1]])
    assert abs(np.corrcoef(embedding[:, 0], embedding[:, 1])[0, 1]) < 0.1
    redundancy = nonlinear_redundancy(embedding, random_state=0)
    assert redundancy[1] > 0.8
    assert redundancy[2] < 0.05


def test_a_single_coordinate_has_zero_redundancy() -> None:
    """With one coordinate there is nothing to predict it from."""
    np.testing.assert_array_equal(
        nonlinear_redundancy(_factors()[:, :1], random_state=0), [0.0]
    )


def test_a_coordinate_with_no_variation_in_its_folds_is_never_negative() -> None:
    """Held-out R^2 below zero is reported as 0."""
    z = _factors()
    assert np.all(nonlinear_redundancy(z, random_state=1) >= 0.0)


def test_the_folds_follow_the_seed() -> None:
    """The same seed gives the same scores."""
    z = _factors()
    embedding = np.column_stack([z[:, 0], z[:, 0] ** 2 - 1])
    np.testing.assert_array_equal(
        nonlinear_redundancy(embedding, random_state=3),
        nonlinear_redundancy(embedding, random_state=3),
    )


def test_one_fold_is_rejected() -> None:
    """Cross-validation needs at least two folds."""
    with pytest.raises(InvalidParameterError, match="n_splits"):
        nonlinear_redundancy(_factors(), n_splits=1)
