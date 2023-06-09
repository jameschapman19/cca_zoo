import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_random_state

from cca_zoo.models import (
    CCA,
    GCCA,
    KCCA,
    KGCCA,
    KTCCA,
    MCCA,
    PLS,
    PLS_ALS,
    TCCA,
    rCCA,
    PCACCA,
)

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


def test_unregularized_methods():
    # This function tests unregularized CCA methods. The idea is that all of these should give the same result.
    latent_dims = 2
    rcca = rCCA(latent_dims=latent_dims).fit([X, Y])
    cca = CCA(latent_dims=latent_dims).fit([X, Y])
    gcca = GCCA(latent_dims=latent_dims).fit([X, Y])
    mcca = MCCA(latent_dims=latent_dims, pca=False).fit([X, Y])
    mcca_pca = MCCA(latent_dims=latent_dims, pca=True).fit([X, Y])
    kcca = KCCA(latent_dims=latent_dims).fit([X, Y])
    kgcca = KGCCA(latent_dims=latent_dims).fit([X, Y])
    tcca = TCCA(latent_dims=latent_dims).fit([X, Y])
    pcacca = PCACCA(latent_dims=latent_dims).fit([X, Y])

    # Get the correlation scores for each method
    corr_rcca = rcca.score((X, Y))
    corr_cca = cca.score((X, Y))
    corr_gcca = gcca.score((X, Y))
    corr_mcca = mcca.score((X, Y))
    corr_mcca_pca = mcca_pca.score((X, Y))
    corr_pcacca = pcacca.score((X, Y))
    corr_kcca = kcca.score((X, Y))
    corr_kgcca = kgcca.score((X, Y))
    corr_tcca = tcca.score((X, Y))

    # Assert that the correlation scores are all equal
    assert np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_tcca, decimal=1) is None
    assert (
        np.testing.assert_array_almost_equal(corr_kgcca, corr_gcca, decimal=1) is None
    )


def test_unregularized_multi():
    # This function tests unregularized CCA methods for more than 2 views. The idea is that all of these should give the same result.
    latent_dims = 2
    gcca = GCCA(latent_dims=latent_dims).fit((X, Y, Z))
    mcca = MCCA(latent_dims=latent_dims).fit((X, Y, Z))
    kcca = KCCA(latent_dims=latent_dims).fit((X, Y, Z))
    # Get the correlation scores for each method
    corr_gcca = gcca.score((X, Y, Z))
    corr_mcca = mcca.score((X, Y, Z))
    corr_kcca = kcca.score((X, Y, Z))

    # Assert that the correlation scores are all equal
    assert np.testing.assert_array_almost_equal(corr_mcca, corr_gcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_mcca, corr_kcca, decimal=1) is None
    # Get the correlation scores for each method
    corr_gcca = gcca.score((X, Y, Z))
    corr_mcca = mcca.score((X, Y, Z))
    corr_kcca = kcca.score((X, Y, Z))

    # Assert that the correlation scores are all equal
    assert np.testing.assert_array_almost_equal(corr_mcca, corr_gcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_mcca, corr_kcca, decimal=1) is None


def test_pls():
    # This function tests PLS and PLS_ALS
    pls_als = PLS_ALS(latent_dims=3, random_state=0)
    pls = PLS(latent_dims=3)

    # Fit both models to the same data
    pls_als.fit([X, Y])
    pls.fit((X, Y))

    # Assert that the scores are close
    pls_score = pls.score((X, Y))
    pls_als_score = pls_als.score((X, Y))

    assert np.allclose(np.abs(pls_als_score), pls_score, rtol=1e-1)


def test_TCCA():
    # This function tests TCCA and KTCCA
    latent_dims = 1
    tcca = TCCA(latent_dims=latent_dims, c=[0.2, 0.2, 0.2]).fit([X, X, Y])
    ktcca = KTCCA(latent_dims=latent_dims, c=[0.2, 0.2]).fit([X, Y])

    # Get the correlation scores for each method
    corr_tcca = tcca.score((X, X, Y))
    corr_ktcca = ktcca.score((X, Y))

    # Assert that the correlation scores are greater than 0.1
    assert corr_tcca > 0.1
    assert corr_ktcca > 0.1
