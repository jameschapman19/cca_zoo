import numpy as np
import pytest
from sklearn.utils.validation import check_random_state

from cca_zoo.linear import CCA, GCCA, MCCA, PCACCA, PLS, PLS_ALS, TCCA, rCCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA


# Setup a fixture for common data
@pytest.fixture
def data():
    n = 10
    rng = check_random_state(0)
    X = rng.rand(n, 3)
    Y = rng.rand(n, 4)
    Z = rng.rand(n, 5)
    # centre the data
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    Z -= Z.mean(axis=0)
    return X, Y, Z


def test_unregularized_methods(data):
    """Test unregularized _CCALoss methods for 2 representations."""
    (
        X,
        Y,
        _,
    ) = data
    latent_dims = 2
    methods = [
        rCCA(latent_dimensions=latent_dims),
        CCA(latent_dimensions=latent_dims),
        KCCA(latent_dimensions=latent_dims),
        PCACCA(latent_dimensions=latent_dims),
        TCCA(latent_dimensions=latent_dims),
        KTCCA(latent_dimensions=latent_dims),
    ]

    scores = [
        method.fit([X, Y]).average_pairwise_correlations((X, Y)) for method in methods
    ]

    # Comparing all scores to the score of the first method (_CCALoss here)
    for score in scores[1:]:
        assert np.testing.assert_array_almost_equal(scores[0], score, decimal=1) is None


def test_unregularized_multi(data):
    """Test unregularized _CCALoss methods for more than 2 representations."""
    X, Y, Z = data
    latent_dims = 2
    methods = [
        GCCA(latent_dimensions=latent_dims),
        MCCA(latent_dimensions=latent_dims),
        KCCA(latent_dimensions=latent_dims),
        GCCA(latent_dimensions=latent_dims),
        MCCA(latent_dimensions=latent_dims, pca=False),
        MCCA(latent_dimensions=latent_dims, pca=True),
        KGCCA(latent_dimensions=latent_dims),
    ]

    scores = [method.fit((X, Y, Z)).score((X, Y, Z)) for method in methods]

    for score in scores[1:]:
        assert np.testing.assert_array_almost_equal(scores[0], score, decimal=1) is None


def test_PLS_methods(data):
    """Test PLS and PLS_ALS methods."""
    X, Y, _ = data
    pls_als = PLS_ALS(latent_dimensions=3, random_state=0)
    pls = PLS(latent_dimensions=3)

    pls_als.fit([X, Y])
    pls.fit((X, Y))

    pls_score = pls.score((X, Y))
    pls_als_score = pls_als.score((X, Y))

    assert np.allclose(np.abs(pls_als_score), pls_score, rtol=1e-1)
