import itertools

import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_random_state

from cca_zoo.models import CCA, PLS, CCA_ALS, SCCA, PMD, ElasticCCA, rCCA, KCCA, KTCCA, MCCA, GCCA, TCCA, SCCA_ADMM, \
    SpanCCA, SWCCA, PLS_ALS, KGCCA

rng = check_random_state(0)
X = rng.rand(500, 20)
Y = rng.rand(500, 21)
Z = rng.rand(500, 22)
X_sp = sp.random(500, 20, density=0.5, random_state=rng)
Y_sp = sp.random(500, 21, density=0.5, random_state=rng)


def test_unregularized_methods():
    # Tests unregularized CCA methods. The idea is that all of these should give the same result.
    latent_dims = 2
    cca = CCA(latent_dims=latent_dims).fit(X, Y)
    iter = CCA_ALS(latent_dims=latent_dims, tol=1e-9, random_state=rng, stochastic=False).fit(X, Y)
    iter_pls = PLS_ALS(latent_dims=latent_dims, tol=1e-9, initialization='unregularized',
                       centre=False).fit(X, Y)
    gcca = GCCA(latent_dims=latent_dims).fit(X, Y)
    mcca = MCCA(latent_dims=latent_dims, eps=1e-9).fit(X, Y)
    kcca = KCCA(latent_dims=latent_dims).fit(X, Y)
    kgcca = KGCCA(latent_dims=latent_dims).fit(X, Y)
    tcca = TCCA(latent_dims=latent_dims).fit(X, Y)
    corr_cca = cca.score(X, Y)
    corr_iter = iter.score(X, Y)
    corr_gcca = gcca.score(X, Y)
    corr_mcca = mcca.score(X, Y)
    corr_kcca = kcca.score(X, Y)
    corr_kgcca = kgcca.score(X, Y)
    corr_tcca = kcca.score(X, Y)
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_tcca, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_kgcca, corr_gcca, decimal=2) is None
    # Check standardized models have standard outputs
    assert np.testing.assert_allclose(np.linalg.norm(iter.transform(X, Y)[0], axis=0) ** 2, 500, rtol=0.1) is None
    assert np.testing.assert_allclose(np.linalg.norm(cca.transform(X, Y)[0], axis=0) ** 2, 500, rtol=0.1) is None
    assert np.testing.assert_allclose(np.linalg.norm(mcca.transform(X, Y)[0], axis=0) ** 2, 500, rtol=0.1) is None
    assert np.testing.assert_allclose(np.linalg.norm(kcca.transform(X, Y)[0], axis=0) ** 2, 500, rtol=0.1) is None
    assert np.testing.assert_allclose(np.linalg.norm(iter.transform(X, Y)[1], axis=0) ** 2, 500, rtol=0.1) is None
    assert np.testing.assert_allclose(np.linalg.norm(cca.transform(X, Y)[1], axis=0) ** 2, 500, rtol=0.1) is None
    assert np.testing.assert_allclose(np.linalg.norm(mcca.transform(X, Y)[1], axis=0) ** 2, 500, rtol=0.1) is None
    assert np.testing.assert_allclose(np.linalg.norm(kcca.transform(X, Y)[1], axis=0) ** 2, 500, rtol=0.1) is None


def test_sparse_input():
    # Tests unregularized CCA methods. The idea is that all of these should give the same result.
    latent_dims = 2
    cca = CCA(latent_dims=latent_dims, centre=False).fit(X_sp, Y_sp)
    iter = CCA_ALS(latent_dims=latent_dims, tol=1e-9, stochastic=False, centre=False).fit(X_sp, Y_sp)
    iter_pls = PLS_ALS(latent_dims=latent_dims, tol=1e-9, initialization='unregularized',
                       centre=False).fit(X_sp, Y_sp)
    gcca = GCCA(latent_dims=latent_dims, centre=False).fit(X_sp, Y_sp)
    mcca = MCCA(latent_dims=latent_dims, centre=False).fit(X_sp, Y_sp)
    kcca = KCCA(latent_dims=latent_dims, centre=False).fit(X_sp, Y_sp)
    scca = SCCA(latent_dims=latent_dims, centre=False, c=0.001).fit(X_sp, Y_sp)
    corr_cca = cca.score(X, Y)
    corr_iter = iter.score(X, Y)
    corr_gcca = gcca.score(X, Y)
    corr_mcca = mcca.score(X, Y)
    corr_kcca = kcca.score(X, Y)
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_iter, corr_mcca, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_iter, corr_gcca, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_iter, corr_kcca, decimal=2) is None


def test_unregularized_multi():
    # Tests unregularized CCA methods for more than 2 views. The idea is that all of these should give the same result.
    latent_dims = 2
    cca = rCCA(latent_dims=latent_dims).fit(X, Y, Z)
    iter = CCA_ALS(latent_dims=latent_dims, stochastic=False, tol=1e-12).fit(X, Y, Z)
    gcca = GCCA(latent_dims=latent_dims).fit(X, Y, Z)
    mcca = MCCA(latent_dims=latent_dims).fit(X, Y, Z)
    kcca = KCCA(latent_dims=latent_dims).fit(X, Y, Z)
    corr_cca = cca.score(X, Y, Z)
    corr_iter = iter.score(X, Y, Z)
    corr_gcca = gcca.score(X, Y, Z)
    corr_mcca = mcca.score(X, Y, Z)
    corr_kcca = kcca.score(X, Y, Z)
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=2) is None
    assert np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=2) is None


def test_regularized_methods():
    # Test that linear regularized methods match PLS solution when using maximum regularisation.
    latent_dims = 2
    c = 1
    kernel = KCCA(latent_dims=latent_dims, c=[c, c], kernel=['linear', 'linear']).fit(X, Y)
    pls = PLS(latent_dims=latent_dims).fit(X, Y)
    gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(X, Y)
    mcca = MCCA(latent_dims=latent_dims, c=[c, c]).fit(X, Y)
    rcca = rCCA(latent_dims=latent_dims, c=[c, c]).fit(X, Y)
    corr_gcca = gcca.score(X, Y)
    corr_mcca = mcca.score(X, Y)
    corr_kernel = kernel.score(X, Y)
    corr_pls = pls.score(X, Y)
    corr_rcca = rcca.score(X, Y)
    # Check the correlations from each unregularized method are the same
    # assert np.testing.assert_array_almost_equal(corr_pls, corr_gcca, decimal=2))
    assert np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_pls, corr_rcca, decimal=1) is None


def test_non_negative_methods():
    latent_dims = 2
    nnelasticca = ElasticCCA(latent_dims=latent_dims, tol=1e-9, positive=True, l1_ratio=[0.5, 0.5],
                             c=[1e-4, 1e-5]).fit(X, Y)
    als = CCA_ALS(latent_dims=latent_dims, tol=1e-9).fit(X, Y)
    nnals = CCA_ALS(latent_dims=latent_dims, tol=1e-9, positive=True).fit(X, Y)
    nnscca = SCCA(latent_dims=latent_dims, tol=1e-9, positive=True, c=[1e-4, 1e-5]).fit(X, Y)


def test_sparse_methods():
    # Test sparsity inducing methods. At the moment just checks running.
    latent_dims = 2
    c1 = [1, 3]
    c2 = [1, 3]

    param_candidates = {'c': list(itertools.product(c1, c2))}
    pmd = PMD(latent_dims=latent_dims, random_state=rng).gridsearch_fit(X, Y,
                                                                        param_candidates=param_candidates,
                                                                        verbose=True, plot=True)
    c1 = [1e-4, 1e-5]
    c2 = [1e-4, 1e-5]
    param_candidates = {'c': list(itertools.product(c1, c2))}
    scca = SCCA(latent_dims=latent_dims, random_state=rng).gridsearch_fit(X, Y,
                                                                          param_candidates=param_candidates,
                                                                          verbose=True)
    elastic = ElasticCCA(latent_dims=latent_dims, random_state=rng).gridsearch_fit(X, Y,
                                                                                   param_candidates=param_candidates,
                                                                                   verbose=True)
    corr_pmd = pmd.score(X, Y)
    corr_scca = scca.score(X, Y)
    corr_elastic = elastic.score(X, Y)
    scca_admm = SCCA_ADMM(c=[1e-4, 1e-4]).fit(X, Y)
    scca = SCCA(c=[1e-4, 1e-4]).fit(X, Y)


def test_weighted_GCCA_methods():
    # Test the 'fancy' additions to GCCA i.e. the view weighting and observation weighting.
    latent_dims = 2
    c = 0
    unweighted_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(X, Y)
    deweighted_gcca = GCCA(latent_dims=latent_dims, c=[c, c], view_weights=[0.5, 0.5]).fit(
        X, Y)
    corr_unweighted_gcca = unweighted_gcca.score(X, Y)
    corr_deweighted_gcca = deweighted_gcca.score(X, Y)
    # Check the correlations from each unregularized method are the same
    K = np.ones((2, X.shape[0]))
    K[0, 200:] = 0
    unobserved_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(X, Y, K=K)
    assert np.testing.assert_array_almost_equal(corr_unweighted_gcca, corr_deweighted_gcca, decimal=1) is None


def test_TCCA():
    # Tests tensor CCA methods
    latent_dims = 2
    tcca = TCCA(latent_dims=latent_dims, c=[0.2, 0.2]).fit(X, Y)
    ktcca = KTCCA(latent_dims=latent_dims, c=[0.2, 0.2]).fit(X, Y)
    corr_tcca = tcca.score(X, Y)
    corr_ktcca = ktcca.score(X, Y)
    assert np.testing.assert_array_almost_equal(corr_tcca, corr_ktcca, decimal=1) is None


def test_cv_fit():
    # Test the CV method
    latent_dims = 2
    c1 = [0.1, 0.2]
    c2 = [0.1, 0.2]
    param_candidates = {'c': list(itertools.product(c1, c2))}
    unweighted_gcca = GCCA(latent_dims=latent_dims).gridsearch_fit(X, Y, folds=5,
                                                                   param_candidates=param_candidates,
                                                                   plot=True, jobs=3)
    deweighted_gcca = GCCA(latent_dims=latent_dims, view_weights=[0.5, 0.5]).gridsearch_fit(
        X, Y, folds=2, param_candidates=param_candidates)
    mcca = MCCA(latent_dims=latent_dims).gridsearch_fit(
        X, Y, folds=2, param_candidates=param_candidates)


def test_l0():
    span_cca = SpanCCA(latent_dims=1, regularisation='l0', c=[2, 2]).fit(X, Y)
    swcca = SWCCA(latent_dims=1, c=[2, 2], sample_support=5).fit(X, Y)
    assert (np.abs(span_cca.weights[0]) > 1e-5).sum() == 2
    assert (np.abs(span_cca.weights[1]) > 1e-5).sum() == 2
    assert (np.abs(swcca.weights[0]) > 1e-5).sum() == 2
    assert (np.abs(swcca.weights[1]) > 1e-5).sum() == 2
    assert (np.abs(swcca.loop.sample_weights) > 1e-5).sum() == 5
