import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state

from cca_zoo.model_selection import (
    cross_validate,
    learning_curve,
    permutation_test_score,
)
from cca_zoo.classical import MPLS

n = 100
rng = check_random_state(0)
X_low = rng.rand(n, 5)
Y_low = rng.rand(n, 5)
n = 5
X_high = rng.rand(n, 100)
Y_high = rng.rand(n, 101)
# centre the data
X_low -= X_low.mean(axis=0)
Y_low -= Y_low.mean(axis=0)
X_high -= X_high.mean(axis=0)
Y_high -= Y_high.mean(axis=0)


def test_explained_variance():
    for X in [X_low, X_high]:
        # Test that explained variance is between 0 and 1
        pls = MPLS(latent_dimensions=5).fit((X, X))
        explained_variance = pls.explained_variance_((X, X))
        explained_variance_ratio = pls.explained_variance_ratio_((X, X))
        explained_variance_cumulative = pls.explained_variance_cumulative_((X, X))
        # explained_variance_ratio should sum to 1 for each view
        assert np.allclose(explained_variance_ratio.sum(axis=1), 1)
        # explained_variance_cumulative should be monotonically increasing
        assert np.all(np.diff(explained_variance_cumulative, axis=1) >= 0)


def test_explained_covariance():
    for X in [X_low, X_high]:
        # Test that explained covariance is between 0 and 1
        pls = MPLS(latent_dimensions=5).fit((X, X))
        explained_covariance = pls.explained_covariance_((X, X))
        explained_covariance_ratio = pls.explained_covariance_ratio_((X, X))
        explained_covariance_cumulative = pls.explained_covariance_cumulative_((X, X))
        # explained_covariance_ratio should sum to 1 for each view
        assert np.allclose(explained_covariance_ratio.sum(), 1)
        # explained_covariance_cumulative should be monotonically increasing
        assert np.all(np.diff(explained_covariance_cumulative) >= 0)


# def test_explained_correlation():


def test_validation():
    # Test that validation works
    pls = MPLS(latent_dimensions=1).fit((X_low, Y_low))
    cross_validate(pls, (X_low, Y_low))
    permutation_test_score(pls, (X_low, Y_low))
    learning_curve(pls, (X_low, Y_low))
