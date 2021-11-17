import numpy as np
import scipy.sparse as sp
from sklearn.utils.fixes import loguniform
from sklearn.utils.validation import check_random_state

from cca_zoo.model_selection import GridSearchCV, RandomizedSearchCV
from cca_zoo.models import (
    rCCA,
    CCA,
    PLS,
    CCA_ALS,
    SCCA,
    PMD,
    ElasticCCA,
    KCCA,
    KTCCA,
    MCCA,
    GCCA,
    TCCA,
    SpanCCA,
    SWCCA,
    PLS_ALS,
    KGCCA,
    NCCA,
    ParkhomenkoCCA,
    SCCA_ADMM,
)
from cca_zoo.utils.plotting import cv_plot

n = 50
rng = check_random_state(0)
X = rng.rand(n, 4)
Y = rng.rand(n, 5)
Z = rng.rand(n, 6)
X_sp = sp.random(n, 4, density=0.5, random_state=rng)
Y_sp = sp.random(n, 5, density=0.5, random_state=rng)


def test_unregularized_methods():
    # Tests unregularized CCA methods. The idea is that all of these should give the same result.
    latent_dims = 2
    cca = CCA(latent_dims=latent_dims).fit([X, Y])
    iter = CCA_ALS(
        latent_dims=latent_dims, tol=1e-9, random_state=rng, stochastic=False
    ).fit([X, Y])
    iter_pls = PLS_ALS(
        latent_dims=latent_dims, tol=1e-9, initialization="unregularized", centre=False
    ).fit([X, Y])
    gcca = GCCA(latent_dims=latent_dims).fit([X, Y])
    mcca = MCCA(latent_dims=latent_dims, eps=1e-9).fit([X, Y])
    kcca = KCCA(latent_dims=latent_dims).fit([X, Y])
    kgcca = KGCCA(latent_dims=latent_dims).fit([X, Y])
    tcca = TCCA(latent_dims=latent_dims).fit([X, Y])
    corr_cca = cca.score((X, Y))
    corr_iter = iter.score((X, Y))
    corr_gcca = gcca.score((X, Y))
    corr_mcca = mcca.score((X, Y))
    corr_kcca = kcca.score((X, Y))
    corr_kgcca = kgcca.score((X, Y))
    corr_tcca = tcca.score((X, Y))
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_tcca, decimal=1) is None
    assert (
        np.testing.assert_array_almost_equal(corr_kgcca, corr_gcca, decimal=1) is None
    )
    # Check standardized models have standard outputs
    assert (
        np.testing.assert_allclose(
            np.linalg.norm(iter.transform((X, Y))[0], axis=0) ** 2, n, rtol=0.2
        )
        is None
    )
    assert (
        np.testing.assert_allclose(
            np.linalg.norm(cca.transform((X, Y))[0], axis=0) ** 2, n, rtol=0.2
        )
        is None
    )
    assert (
        np.testing.assert_allclose(
            np.linalg.norm(mcca.transform((X, Y))[0], axis=0) ** 2, n, rtol=0.2
        )
        is None
    )
    assert (
        np.testing.assert_allclose(
            np.linalg.norm(kcca.transform((X, Y))[0], axis=0) ** 2, n, rtol=0.2
        )
        is None
    )
    assert (
        np.testing.assert_allclose(
            np.linalg.norm(iter.transform((X, Y))[1], axis=0) ** 2, n, rtol=0.2
        )
        is None
    )
    assert (
        np.testing.assert_allclose(
            np.linalg.norm(cca.transform((X, Y))[1], axis=0) ** 2, n, rtol=0.2
        )
        is None
    )
    assert (
        np.testing.assert_allclose(
            np.linalg.norm(mcca.transform((X, Y))[1], axis=0) ** 2, n, rtol=0.2
        )
        is None
    )
    assert (
        np.testing.assert_allclose(
            np.linalg.norm(kcca.transform((X, Y))[1], axis=0) ** 2, n, rtol=0.2
        )
        is None
    )


def test_sparse_input():
    # Tests unregularized CCA methods. The idea is that all of these should give the same result.
    latent_dims = 2
    cca = CCA(latent_dims=latent_dims, centre=False).fit((X_sp, Y_sp))
    iter = CCA_ALS(
        latent_dims=latent_dims, tol=1e-9, stochastic=False, centre=False
    ).fit((X_sp, Y_sp))
    iter_pls = PLS_ALS(
        latent_dims=latent_dims, tol=1e-9, initialization="unregularized", centre=False
    ).fit((X_sp, Y_sp))
    gcca = GCCA(latent_dims=latent_dims, centre=False).fit((X_sp, Y_sp))
    mcca = MCCA(latent_dims=latent_dims, centre=False).fit((X_sp, Y_sp))
    kcca = KCCA(latent_dims=latent_dims, centre=False).fit((X_sp, Y_sp))
    scca = SCCA(latent_dims=latent_dims, centre=False, c=0.001).fit((X_sp, Y_sp))
    corr_cca = cca.score((X, Y))
    corr_iter = iter.score((X, Y))
    corr_gcca = gcca.score((X, Y))
    corr_mcca = mcca.score((X, Y))
    corr_kcca = kcca.score((X, Y))
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_iter, corr_mcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_iter, corr_gcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_iter, corr_kcca, decimal=1) is None


def test_unregularized_multi():
    # Tests unregularized CCA methods for more than 2 views. The idea is that all of these should give the same result.
    latent_dims = 2
    cca = rCCA(latent_dims=latent_dims).fit((X, Y, Z))
    iter = CCA_ALS(latent_dims=latent_dims, stochastic=False, tol=1e-12).fit((X, Y, Z))
    gcca = GCCA(latent_dims=latent_dims).fit((X, Y, Z))
    mcca = MCCA(latent_dims=latent_dims).fit((X, Y, Z))
    kcca = KCCA(latent_dims=latent_dims).fit((X, Y, Z))
    corr_cca = cca.score((X, Y, Z))
    corr_iter = iter.score((X, Y, Z))
    corr_gcca = gcca.score((X, Y, Z))
    corr_mcca = mcca.score((X, Y, Z))
    corr_kcca = kcca.score((X, Y, Z))
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=1) is None
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


def test_non_negative_methods():
    latent_dims = 2
    nnscca = SCCA(
        latent_dims=latent_dims, tol=1e-9, positive=True, c=[1e-1, 1e-1], random_state=0
    ).fit((X, Y))
    nnelastic = ElasticCCA(
        latent_dims=latent_dims,
        tol=1e-9,
        positive=True,
        l1_ratio=[0.5, 0.5],
        c=[1e-4, 1e-5],
        random_state=0,
    ).fit([X, Y])
    nnals = CCA_ALS(
        latent_dims=latent_dims, tol=1e-9, positive=True, random_state=0
    ).fit([X, Y])
    assert np.testing.assert_array_less(-1e-9, nnelastic.weights[0]) is None
    assert np.testing.assert_array_less(-1e-9, nnelastic.weights[1]) is None
    assert np.testing.assert_array_less(-1e-9, nnals.weights[0]) is None
    assert np.testing.assert_array_less(-1e-9, nnals.weights[1]) is None
    assert np.testing.assert_array_less(-1e-9, nnscca.weights[0]) is None
    assert np.testing.assert_array_less(-1e-9, nnscca.weights[1]) is None


def test_sparse_methods():
    c1 = [1, 3]
    c2 = [1, 3]
    param_grid = {"c": [c1, c2]}
    pmd_cv = GridSearchCV(PMD(random_state=rng), param_grid=param_grid).fit([X, Y])
    cv_plot(pmd_cv.cv_results_)
    c1 = [5e-1]
    c2 = [1e-1]
    param_grid = {"c": [c1, c2]}
    scca_cv = GridSearchCV(SCCA(random_state=rng), param_grid=param_grid).fit([X, Y])
    c1 = [1e-1]
    c2 = [1e-1]
    param_grid = {"c": [c1, c2]}
    parkhomenko_cv = GridSearchCV(
        ParkhomenkoCCA(random_state=rng), param_grid=param_grid
    ).fit([X, Y])
    c1 = [2e-2]
    c2 = [1e-2]
    param_grid = {"c": [c1, c2]}
    admm_cv = GridSearchCV(SCCA_ADMM(random_state=rng), param_grid=param_grid).fit(
        [X, Y]
    )
    c1 = loguniform(1e-1, 2e-1)
    c2 = loguniform(1e-1, 2e-1)
    param_grid = {"c": [c1, c2], "l1_ratio": [[0.9], [0.9]]}
    elastic_cv = RandomizedSearchCV(
        ElasticCCA(random_state=rng), param_distributions=param_grid, n_iter=4
    ).fit([X, Y])
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
    assert corr_tcca > 0.4
    assert corr_ktcca > 0.4


def test_NCCA():
    latent_dims = 1
    ncca = NCCA(latent_dims=latent_dims).fit((X, Y))
    corr_ncca = ncca.score((X, Y))
    assert corr_ncca > 0.9


def test_l0():
    span_cca = SpanCCA(latent_dims=1, regularisation="l0", c=[2, 2]).fit([X, Y])
    swcca = SWCCA(latent_dims=1, c=[2, 2], sample_support=5).fit([X, Y])
    assert (np.abs(span_cca.weights[0]) > 1e-5).sum() == 2
    assert (np.abs(span_cca.weights[1]) > 1e-5).sum() == 2
    assert (np.abs(swcca.weights[0]) > 1e-5).sum() == 2
    assert (np.abs(swcca.weights[1]) > 1e-5).sum() == 2
    assert (np.abs(swcca.loop.sample_weights) > 1e-5).sum() == 5


def test_VCCA():
    try:
        from cca_zoo.probabilisticmodels import VariationalCCA
        from cca_zoo.data import generate_simple_data

        # Tests tensor CCA methods
        (X, Y), (_) = generate_simple_data(20, [9, 9], random_state=rng, eps=0.1)
        latent_dims = 1
        cca = CCA(latent_dims=latent_dims).fit([X, Y])
        vcca = VariationalCCA(
            latent_dims=latent_dims, num_warmup=500, num_samples=500
        ).fit([X, Y])
        # Test that vanilla CCA and VCCA produce roughly similar latent space
        assert (
            np.corrcoef(
                cca.transform([X, Y])[1].T,
                vcca.posterior_samples["z"].mean(axis=0)[:, 0],
            )[0, 1]
            > 0.9
        )
    except:
        # some might not have access to jax/numpyro so leave this as an optional test locally.
        pass
