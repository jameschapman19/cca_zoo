import itertools
from unittest import TestCase

import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_random_state

from cca_zoo.models import CCA, PLS, CCA_ALS, SCCA, PMD, ElasticCCA, rCCA, KCCA, KTCCA, MCCA, GCCA, TCCA, SCCA_ADMM, \
    SpanCCA, SWCCA


class TestModels(TestCase):

    def setUp(self):
        self.rng = check_random_state(0)
        self.X = self.rng.rand(500, 20)
        self.Y = self.rng.rand(500, 21)
        self.Z = self.rng.rand(500, 22)
        self.X_sp = sp.random(500, 20, density=0.5, random_state=self.rng)
        self.Y_sp = sp.random(500, 21, density=0.5, random_state=self.rng)

    def tearDown(self):
        pass

    def test_unregularized_methods(self):
        # Tests unregularized CCA methods. The idea is that all of these should give the same result.
        latent_dims = 2
        cca = CCA(latent_dims=latent_dims).fit(self.X, self.Y)
        iter = CCA_ALS(latent_dims=latent_dims, tol=1e-9, random_state=self.rng, stochastic=False).fit(self.X, self.Y)
        gcca = GCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        mcca = MCCA(latent_dims=latent_dims, eps=1e-9).fit(self.X, self.Y)
        kcca = KCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        tcca = TCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        corr_cca = cca.score(self.X, self.Y)
        corr_iter = iter.score(self.X, self.Y)
        corr_gcca = gcca.score(self.X, self.Y)
        corr_mcca = mcca.score(self.X, self.Y)
        corr_kcca = kcca.score(self.X, self.Y)
        corr_tcca = kcca.score(self.X, self.Y)
        # Check the score outputs are the right shape
        self.assertTrue(iter.scores[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(gcca.scores[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(mcca.scores[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(kcca.scores[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(tcca.scores[0].shape == (self.X.shape[0], latent_dims))
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_tcca, decimal=2))
        # Check standardized models have standard outputs
        self.assertIsNone(np.testing.assert_allclose(np.linalg.norm(iter.scores[0], axis=0) ** 2, 500))
        self.assertIsNone(np.testing.assert_allclose(np.linalg.norm(cca.scores[0], axis=0) ** 2, 500))
        self.assertIsNone(np.testing.assert_allclose(np.linalg.norm(mcca.scores[0], axis=0) ** 2, 500, rtol=0.1))
        self.assertIsNone(np.testing.assert_allclose(np.linalg.norm(kcca.scores[0], axis=0) ** 2, 500, rtol=0.1))
        self.assertIsNone(np.testing.assert_allclose(np.linalg.norm(iter.scores[1], axis=0) ** 2, 500))
        self.assertIsNone(np.testing.assert_allclose(np.linalg.norm(cca.scores[1], axis=0) ** 2, 500))
        self.assertIsNone(np.testing.assert_allclose(np.linalg.norm(mcca.scores[1], axis=0) ** 2, 500, rtol=0.1))
        self.assertIsNone(np.testing.assert_allclose(np.linalg.norm(kcca.scores[1], axis=0) ** 2, 500, rtol=0.1))

    def test_sparse_input(self):
        # Tests unregularized CCA methods. The idea is that all of these should give the same result.
        latent_dims = 2
        cca = CCA(latent_dims=latent_dims, centre=False).fit(self.X_sp, self.Y_sp)
        iter = CCA_ALS(latent_dims=latent_dims, tol=1e-9, stochastic=False, centre=False,
                       initialization='unregularized').fit(self.X_sp, self.Y_sp)
        gcca = GCCA(latent_dims=latent_dims, centre=False).fit(self.X_sp, self.Y_sp)
        mcca = MCCA(latent_dims=latent_dims, centre=False).fit(self.X_sp, self.Y_sp)
        kcca = KCCA(latent_dims=latent_dims, centre=False).fit(self.X_sp, self.Y_sp)
        scca = SCCA(latent_dims=latent_dims, centre=False, c=0.001).fit(self.X_sp, self.Y_sp)
        corr_cca = cca.score(self.X, self.Y)
        corr_iter = iter.score(self.X, self.Y)
        corr_gcca = gcca.score(self.X, self.Y)
        corr_mcca = mcca.score(self.X, self.Y)
        corr_kcca = kcca.score(self.X, self.Y)
        # Check the score outputs are the right shape
        self.assertTrue(iter.scores[0].shape == (self.X_sp.shape[0], latent_dims))
        self.assertTrue(gcca.scores[0].shape == (self.X_sp.shape[0], latent_dims))
        self.assertTrue(mcca.scores[0].shape == (self.X_sp.shape[0], latent_dims))
        self.assertTrue(kcca.scores[0].shape == (self.X_sp.shape[0], latent_dims))
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_mcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_kcca, decimal=2))

    def test_unregularized_multi(self):
        # Tests unregularized CCA methods for more than 2 views. The idea is that all of these should give the same result.
        latent_dims = 2
        cca = rCCA(latent_dims=latent_dims).fit(self.X, self.Y, self.Z)
        iter = CCA_ALS(latent_dims=latent_dims, stochastic=False, tol=1e-12).fit(self.X, self.Y, self.Z)
        gcca = GCCA(latent_dims=latent_dims).fit(self.X, self.Y, self.Z)
        mcca = MCCA(latent_dims=latent_dims).fit(self.X, self.Y, self.Z)
        kcca = KCCA(latent_dims=latent_dims).fit(self.X, self.Y, self.Z)
        corr_cca = cca.score(self.X, self.Y, self.Z)
        corr_iter = iter.score(self.X, self.Y, self.Z)
        corr_gcca = gcca.score(self.X, self.Y, self.Z)
        corr_mcca = mcca.score(self.X, self.Y, self.Z)
        corr_kcca = kcca.score(self.X, self.Y, self.Z)
        # Check the score outputs are the right shape
        self.assertTrue(iter.scores[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(gcca.scores[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(mcca.scores[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(kcca.scores[0].shape == (self.X.shape[0], latent_dims))
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=2))

    def test_regularized_methods(self):
        # Test that linear regularized methods match PLS solution when using maximum regularisation.
        latent_dims = 2
        c = 1
        kernel = KCCA(latent_dims=latent_dims, c=[c, c], kernel=['linear', 'linear']).fit(self.X,
                                                                                          self.Y)
        pls = PLS(latent_dims=latent_dims).fit(self.X, self.Y)
        gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        mcca = MCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        rcca = rCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        corr_gcca = gcca.score(self.X, self.Y)
        corr_mcca = mcca.score(self.X, self.Y)
        corr_kernel = kernel.score(self.X, self.Y)
        corr_pls = pls.score(self.X, self.Y)
        corr_rcca = rcca.score(self.X, self.Y)
        # Check the correlations from each unregularized method are the same
        # self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_rcca, decimal=1))

    def test_non_negative_methods(self):
        latent_dims = 2
        nnelasticca = ElasticCCA(latent_dims=latent_dims, tol=1e-9, positive=True, l1_ratio=[0.5, 0.5],
                                 c=[1e-4, 1e-5]).fit(self.X, self.Y)
        als = CCA_ALS(latent_dims=latent_dims, tol=1e-9).fit(self.X, self.Y)
        nnals = CCA_ALS(latent_dims=latent_dims, tol=1e-9, positive=True).fit(self.X, self.Y)
        nnscca = SCCA(latent_dims=latent_dims, tol=1e-9, positive=True, c=[1e-4, 1e-5]).fit(self.X, self.Y)

    def test_sparse_methods(self):
        # Test sparsity inducing methods. At the moment just checks running.
        latent_dims = 2
        c1 = [1, 3]
        c2 = [1, 3]

        param_candidates = {'c': list(itertools.product(c1, c2))}
        pmd = PMD(latent_dims=latent_dims, random_state=self.rng).gridsearch_fit(self.X, self.Y,
                                                                                 param_candidates=param_candidates,
                                                                                 verbose=True, plot=True)
        c1 = [1e-4, 1e-5]
        c2 = [1e-4, 1e-5]
        param_candidates = {'c': list(itertools.product(c1, c2))}
        scca = SCCA(latent_dims=latent_dims, random_state=self.rng).gridsearch_fit(self.X, self.Y,
                                                                                   param_candidates=param_candidates,
                                                                                   verbose=True)
        elastic = ElasticCCA(latent_dims=latent_dims, random_state=self.rng).gridsearch_fit(self.X, self.Y,
                                                                                            param_candidates=param_candidates,
                                                                                            verbose=True)
        corr_pmd = pmd.score(self.X, self.Y)
        corr_scca = scca.score(self.X, self.Y)
        corr_elastic = elastic.score(self.X, self.Y)
        scca_admm = SCCA_ADMM(c=[1e-4, 1e-4]).fit(self.X, self.Y)
        scca = SCCA(c=[1e-4, 1e-4]).fit(self.X, self.Y)

    def test_weighted_GCCA_methods(self):
        # Test the 'fancy' additions to GCCA i.e. the view weighting and observation weighting.
        latent_dims = 2
        c = 0
        unweighted_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        deweighted_gcca = GCCA(latent_dims=latent_dims, c=[c, c], view_weights=[0.5, 0.5]).fit(
            self.X, self.Y)
        corr_unweighted_gcca = unweighted_gcca.score(self.X, self.Y)
        corr_deweighted_gcca = deweighted_gcca.score(self.X, self.Y)
        # Check the correlations from each unregularized method are the same
        K = np.ones((2, self.X.shape[0]))
        K[0, 200:] = 0
        unobserved_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y, K=K)
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_unweighted_gcca, corr_deweighted_gcca, decimal=1))

    def test_TCCA(self):
        # Tests tensor CCA methods
        latent_dims = 2
        tcca = TCCA(latent_dims=latent_dims, c=[0.2, 0.2]).fit(self.X, self.Y)
        ktcca = KTCCA(latent_dims=latent_dims, c=[0.2, 0.2]).fit(self.X, self.Y)
        corr_tcca = tcca.score(self.X, self.Y)
        corr_ktcca = ktcca.score(self.X, self.Y)
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_tcca, corr_ktcca, decimal=1))

    def test_cv_fit(self):
        # Test the CV method
        latent_dims = 2
        c1 = [0.1, 0.2]
        c2 = [0.1, 0.2]
        param_candidates = {'c': list(itertools.product(c1, c2))}
        unweighted_gcca = GCCA(latent_dims=latent_dims).gridsearch_fit(self.X, self.Y, folds=5,
                                                                       param_candidates=param_candidates,
                                                                       plot=True, jobs=3)
        deweighted_gcca = GCCA(latent_dims=latent_dims, view_weights=[0.5, 0.5]).gridsearch_fit(
            self.X, self.Y, folds=2, param_candidates=param_candidates)
        mcca = MCCA(latent_dims=latent_dims).gridsearch_fit(
            self.X, self.Y, folds=2, param_candidates=param_candidates)

    def test_l0(self):
        span_cca = SpanCCA(latent_dims=1, regularisation='l0', c=[2, 2]).fit(self.X, self.Y)
        swcca = SWCCA(latent_dims=1, c=[2, 2], sample_support=5).fit(self.X, self.Y)
        self.assertEqual((np.abs(span_cca.weights[0]) > 1e-5).sum(), 2)
        self.assertEqual((np.abs(span_cca.weights[1]) > 1e-5).sum(), 2)
        self.assertEqual((np.abs(swcca.weights[0]) > 1e-5).sum(), 2)
        self.assertEqual((np.abs(swcca.weights[1]) > 1e-5).sum(), 2)
        self.assertEqual((np.abs(swcca.loop.sample_weights) > 1e-5).sum(), 5)
        print()
