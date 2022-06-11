import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.utils.fixes import loguniform
from sklearn.utils.validation import check_random_state

from cca_zoo import cross_validate, permutation_test_score, learning_curve
from cca_zoo.data import generate_covariance_data
from cca_zoo.model_selection import GridSearchCV, RandomizedSearchCV
from cca_zoo.models import (
    rCCA,
    CCA,
    PLS,
    SCCA_IPLS,
    SCCA_PMD,
    ElasticCCA,
    KCCA,
    KTCCA,
    MCCA,
    GCCA,
    TCCA,
    SCCA_Span,
    SWCCA,
    KGCCA,
    NCCA,
    PartialCCA,
    PLS_ALS,
    SCCA_ADMM,
    SCCA_Parkhomenko,
    StochasticPowerPLS,
    IncrementalPLS,
)
from cca_zoo.plotting import pairplot_train_test

n = 50
rng = check_random_state(0)
X = rng.rand(n, 10)
Y = rng.rand(n, 11)
Z = rng.rand(n, 12)
X_sp = sp.random(n, 10, density=0.5, random_state=rng)
Y_sp = sp.random(n, 11, density=0.5, random_state=rng)


def test_unregularized_methods():
    # Tests unregularized CCA methods. The idea is that all of these should give the same result.
    latent_dims = 2
    cca = CCA(latent_dims=latent_dims).fit([X, Y])
    gcca = GCCA(latent_dims=latent_dims).fit([X, Y])
    mcca = MCCA(latent_dims=latent_dims).fit([X, Y])
    kcca = KCCA(latent_dims=latent_dims).fit([X, Y])
    kgcca = KGCCA(latent_dims=latent_dims).fit([X, Y])
    tcca = TCCA(latent_dims=latent_dims).fit([X, Y])
    corr_cca = cca.score((X, Y))
    corr_gcca = gcca.score((X, Y))
    corr_mcca = mcca.score((X, Y))
    corr_kcca = kcca.score((X, Y))
    corr_kgcca = kgcca.score((X, Y))
    corr_tcca = tcca.score((X, Y))
    assert np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_tcca, decimal=1) is None
    assert (
        np.testing.assert_array_almost_equal(corr_kgcca, corr_gcca, decimal=1) is None
    )


def test_unregularized_multi():
    # Tests unregularized CCA methods for more than 2 views. The idea is that all of these should give the same result.
    latent_dims = 2
    cca = rCCA(latent_dims=latent_dims).fit((X, Y, Z))
    gcca = GCCA(latent_dims=latent_dims).fit((X, Y, Z))
    mcca = MCCA(latent_dims=latent_dims).fit((X, Y, Z))
    kcca = KCCA(latent_dims=latent_dims).fit((X, Y, Z))
    corr_cca = cca.score((X, Y, Z))
    corr_gcca = gcca.score((X, Y, Z))
    corr_mcca = mcca.score((X, Y, Z))
    corr_kcca = kcca.score((X, Y, Z))
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=1) is None


def test_regularized_methods():
    # Test that linear regularized methods match PLS solution when using maximum regularisation.
    latent_dims = 2
    c = 1
    kernel = KCCA(latent_dims=latent_dims, c=[c, c], kernel=["linear", "linear"]).fit(
        (X, Y)
    )
    pls = PLS(latent_dims=latent_dims).fit([X, Y])
    gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit([X, Y])
    mcca = MCCA(latent_dims=latent_dims, c=[c, c]).fit([X, Y])
    rcca = rCCA(latent_dims=latent_dims, c=[c, c]).fit([X, Y])
    corr_gcca = gcca.score((X, Y))
    corr_mcca = mcca.score((X, Y))
    corr_kernel = kernel.score((X, Y))
    corr_pls = pls.score((X, Y))
    corr_rcca = rcca.score((X, Y))
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=1) is None
    assert (
        np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=1) is None
    )
    assert np.testing.assert_array_almost_equal(corr_pls, corr_rcca, decimal=1) is None


def test_sparse_methods():
    c1 = loguniform(1e-1, 2e-1)
    c2 = loguniform(1e-1, 2e-1)
    param_grid = {"c": [c1, c2], "l1_ratio": [[0.9], [0.9]]}
    elastic_cv = RandomizedSearchCV(
        ElasticCCA(random_state=rng), param_distributions=param_grid, n_iter=4
    ).fit([X, Y])
    c1 = [1e-1]
    c2 = [1e-1]
    param_grid = {"c": [c1, c2]}
    scca_cv = GridSearchCV(SCCA_IPLS(random_state=rng), param_grid=param_grid).fit(
        [X, Y]
    )
    c1 = [0.1, 0.2]
    c2 = [0.1, 0.2]
    param_grid = {"c": [c1, c2]}
    pmd_cv = GridSearchCV(SCCA_PMD(random_state=rng), param_grid=param_grid).fit([X, Y])
    c1 = [1e-1]
    c2 = [1e-1]
    param_grid = {"c": [c1, c2]}
    parkhomenko_cv = GridSearchCV(
        SCCA_Parkhomenko(random_state=rng), param_grid=param_grid
    ).fit([X, Y])
    c1 = [2e-2]
    c2 = [1e-2]
    param_grid = {"c": [c1, c2]}
    admm_cv = GridSearchCV(SCCA_ADMM(random_state=rng), param_grid=param_grid).fit(
        [X, Y]
    )
    assert (pmd_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (pmd_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (scca_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (scca_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (admm_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (admm_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (parkhomenko_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (parkhomenko_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (elastic_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (elastic_cv.best_estimator_.weights[1] == 0).sum() > 0


def test_pls():
    pls_als = PLS_ALS(latent_dims=3)
    pls = PLS(latent_dims=3)
    pls_als.fit((X, Y))
    pls.fit((X, Y))
    assert np.allclose(np.abs(pls_als.weights[0]), np.abs(pls.weights[0]), rtol=1e-1)


def test_weighted_GCCA_methods():
    # TODO we have view weighted GCCA and missing observation GCCA
    latent_dims = 2
    c = 0
    unweighted_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit([X, Y])
    deweighted_gcca = GCCA(
        latent_dims=latent_dims, c=[c, c], view_weights=[0.5, 0.5]
    ).fit([X, Y])
    corr_unweighted_gcca = unweighted_gcca.score((X, Y))
    corr_deweighted_gcca = deweighted_gcca.score((X, Y))
    # Check the correlations from each unregularized method are the same
    K = np.ones((2, X.shape[0]))
    K[0, 200:] = 0
    unobserved_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit((X, Y), K=K)
    assert (
        np.testing.assert_array_almost_equal(
            corr_unweighted_gcca, corr_deweighted_gcca, decimal=1
        )
        is None
    )


def test_TCCA():
    latent_dims = 1
    tcca = TCCA(latent_dims=latent_dims, c=[0.2, 0.2, 0.2]).fit([X, X, Y])
    ktcca = KTCCA(latent_dims=latent_dims, c=[0.2, 0.2]).fit([X, Y])
    corr_tcca = tcca.score((X, X, Y))
    corr_ktcca = ktcca.score((X, Y))
    assert corr_tcca > 0.1
    assert corr_ktcca > 0.1


def test_NCCA():
    latent_dims = 1
    ncca = NCCA(latent_dims=latent_dims).fit((X, Y))
    corr_ncca = ncca.score((X, Y))
    assert corr_ncca > 0.9


def test_l0():
    span_cca = SCCA_Span(
        latent_dims=1, regularisation="l0", c=[2, 2], random_state=rng
    ).fit([X, Y])
    swcca = SWCCA(latent_dims=1, c=[5, 5], sample_support=5, random_state=rng).fit(
        [X, Y]
    )
    assert (np.abs(span_cca.weights[0]) > 1e-5).sum() == 2
    assert (np.abs(span_cca.weights[1]) > 1e-5).sum() == 2
    assert (np.abs(swcca.weights[0]) > 1e-5).sum() == 5
    assert (np.abs(swcca.weights[1]) > 1e-5).sum() == 5
    assert (np.abs(swcca.loop.sample_weights) > 1e-5).sum() == 5


def test_partialcca():
    # Tests that partial CCA scores are not correlated with partials
    pcca = PartialCCA(latent_dims=3)
    pcca.fit((X, Y), partials=Z)
    assert np.allclose(
        np.corrcoef(pcca.transform((X, Y), partials=Z)[0], Z, rowvar=False)[:3, :3]
        - np.eye(3),
        0,
        atol=0.001,
    )


def test_stochastic():
    pls = PLS(latent_dims=1).fit((X, Y))
    ipls = IncrementalPLS(latent_dims=1, epochs=10, simple=False, batch_size=1).fit(
        (X, Y)
    )
    spls = StochasticPowerPLS(latent_dims=1, epochs=10, batch_size=1, lr=1e-2).fit(
        (X, Y)
    )
    pls_score = pls.score((X, Y))
    ipls_score = ipls.score((X, Y))
    spls_score = spls.score((X, Y))
    assert (
        np.testing.assert_array_almost_equal(pls_score, ipls_score, decimal=1) is None
    )
    assert (
        np.testing.assert_array_almost_equal(pls_score, spls_score, decimal=1) is None
    )


def test_validation():
    # Test that validation works
    pls = PLS(latent_dims=1).fit((X, Y))
    cross_validate(pls, (X, Y))
    permutation_test_score(pls, (X, Y))
    learning_curve(pls, (X, Y))


def test_plotting():
    pls = PLS(latent_dims=1).fit((X, Y))
    X_te = np.random.rand(*X.shape)
    Y_te = np.random.rand(*Y.shape)
    pairplot_train_test(pls.transform((X, Y)), pls.transform((X_te, Y_te)))


def test_PCCA():
    # some might not have access to jax/numpyro so leave this as an optional test locally.
    numpyro = pytest.importorskip("numpyro")
    from cca_zoo.probabilisticmodels import ProbabilisticCCA

    np.random.seed(0)
    # Tests tensor CCA methods
    (X, Y), (_) = generate_covariance_data(20, [5, 5])
    latent_dims = 1
    cca = CCA(latent_dims=latent_dims).fit([X, Y])
    pcca = ProbabilisticCCA(
        latent_dims=latent_dims, num_warmup=1000, num_samples=1000
    ).fit([X, Y])
    # Test that vanilla CCA and VCCA produce roughly similar latent space ie they are correlated
    assert (
        np.abs(
            np.corrcoef(
                cca.transform([X, Y])[1].T,
                pcca.posterior_samples["z"].mean(axis=0)[:, 0],
            )[0, 1]
        )
        > 0.9
    )
