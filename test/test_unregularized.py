import pytest
import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_random_state
from cca_zoo.linear import CCA, GCCA, MCCA, PCACCA, PLS, PLS_ALS, TCCA, rCCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA


# Setup a fixture for common data
@pytest.fixture
def data():
    n = 50
    rng = check_random_state(0)
    X = rng.rand(n, 11)
    Y = rng.rand(n, 10)
    Z = rng.rand(n, 12)
    X_sp = sp.random(n, 10, density=0.5, random_state=rng)
    Y_sp = sp.random(n, 11, density=0.5, random_state=rng)
    # centre the data
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    Z -= Z.mean(axis=0)
    X_sp -= X_sp.mean(axis=0)
    Y_sp -= Y_sp.mean(axis=0)
    return X, Y, Z, X_sp, Y_sp


def test_unregularized_methods(data):
    """Test unregularized CCA methods for 2 views."""
    X, Y, _, _, _ = data
    latent_dims = 2
    methods = [
        rCCA(latent_dimensions=latent_dims),
        CCA(latent_dimensions=latent_dims),
        GCCA(latent_dimensions=latent_dims),
        MCCA(latent_dimensions=latent_dims, pca=False),
        MCCA(latent_dimensions=latent_dims, pca=True),
        KCCA(latent_dimensions=latent_dims),
        KGCCA(latent_dimensions=latent_dims),
        TCCA(latent_dimensions=latent_dims),
        PCACCA(latent_dimensions=latent_dims),
    ]

    scores = [method.fit([X, Y]).score((X, Y)) for method in methods]

    # Comparing all scores to the score of the first method (CCA here)
    for score in scores[1:]:
        assert np.testing.assert_array_almost_equal(scores[0], score, decimal=1) is None


def test_unregularized_multi(data):
    """Test unregularized CCA methods for more than 2 views."""
    X, Y, Z, _, _ = data
    latent_dims = 2
    methods = [
        GCCA(latent_dimensions=latent_dims),
        MCCA(latent_dimensions=latent_dims),
        KCCA(latent_dimensions=latent_dims),
    ]

    scores = [method.fit((X, Y, Z)).score((X, Y, Z)) for method in methods]

    for score in scores[1:]:
        assert np.testing.assert_array_almost_equal(scores[0], score, decimal=1) is None


def test_PLS_methods(data):
    """Test PLS and PLS_ALS methods."""
    X, Y, _, _, _ = data
    pls_als = PLS_ALS(latent_dimensions=3, random_state=0)
    pls = PLS(latent_dimensions=3)

    pls_als.fit([X, Y])
    pls.fit((X, Y))

    pls_score = pls.score((X, Y))
    pls_als_score = pls_als.score((X, Y))

    assert np.allclose(np.abs(pls_als_score), pls_score, rtol=1e-1)


def test_TCCA_methods(data):
    """Test TCCA and KTCCA methods."""
    X, Y, _, _, _ = data
    latent_dims = 1
    tcca = TCCA(latent_dimensions=latent_dims, c=[0.2, 0.2, 0.2]).fit([X, X, Y])
    ktcca = KTCCA(latent_dimensions=latent_dims, c=[0.2, 0.2]).fit([X, Y])

    corr_tcca = tcca.score((X, X, Y))
    corr_ktcca = ktcca.score((X, Y))

    assert corr_tcca > 0.1
    assert corr_ktcca > 0.1
