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
        self.X_sp = sp.random(1000, 20, density=0.1, random_state=self.rng)
        self.Y_sp = sp.random(1000, 21, density=0.1, random_state=self.rng)

    def tearDown(self):
        pass

    def test_unregularized_methods(self):
        # Tests unregularized CCA methods. The idea is that all of these should give the same result.
        latent_dims = 1
        wrap_cca = CCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_iter = CCA_ALS(latent_dims=latent_dims, tol=1e-9, random_state=self.rng).fit(self.X, self.Y)
        wrap_gcca = GCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_mcca = MCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_kcca = KCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        corr_cca = wrap_cca.train_correlations[0, 1]
        corr_iter = wrap_iter.train_correlations[0, 1]
        corr_gcca = wrap_gcca.train_correlations[0, 1]
        corr_mcca = wrap_mcca.train_correlations[0, 1]
        corr_kcca = wrap_kcca.train_correlations[0, 1]
        # Check the score outputs are the right shape
        self.assertTrue(wrap_iter.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_gcca.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_mcca.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_kcca.score_list[0].shape == (self.X.shape[0], latent_dims))
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_mcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_kcca, decimal=2))

    def test_sparse_input(self):
        # Tests unregularized CCA methods. The idea is that all of these should give the same result.
        latent_dims = 1
        wrap_cca = CCA(latent_dims=latent_dims).fit(self.X_sp, self.Y_sp)
        wrap_iter = CCA_ALS(latent_dims=latent_dims, tol=1e-9).fit(self.X_sp, self.Y_sp)
        wrap_gcca = GCCA(latent_dims=latent_dims).fit(self.X_sp, self.Y_sp)
        wrap_mcca = MCCA(latent_dims=latent_dims).fit(self.X_sp, self.Y_sp)
        wrap_kcca = KCCA(latent_dims=latent_dims).fit(self.X_sp, self.Y_sp)
        corr_cca = wrap_cca.train_correlations[0, 1]
        corr_iter = wrap_iter.train_correlations[0, 1]
        corr_gcca = wrap_gcca.train_correlations[0, 1]
        corr_mcca = wrap_mcca.train_correlations[0, 1]
        corr_kcca = wrap_kcca.train_correlations[0, 1]
        # Check the score outputs are the right shape
        self.assertTrue(wrap_iter.score_list[0].shape == (self.X_sp.shape[0], latent_dims))
        self.assertTrue(wrap_gcca.score_list[0].shape == (self.X_sp.shape[0], latent_dims))
        self.assertTrue(wrap_mcca.score_list[0].shape == (self.X_sp.shape[0], latent_dims))
        self.assertTrue(wrap_kcca.score_list[0].shape == (self.X_sp.shape[0], latent_dims))
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_mcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_kcca, decimal=2))

    def test_unregularized_multi(self):
        # Tests unregularized CCA methods for more than 2 views. The idea is that all of these should give the same result.
        latent_dims = 5
        wrap_cca = rCCA(latent_dims=latent_dims).fit(self.X, self.Y, self.Z)
        wrap_iter = CCA_ALS(latent_dims=latent_dims, stochastic=False, tol=1e-12).fit(self.X, self.Y,
                                                                                      self.Z)
        wrap_gcca = GCCA(latent_dims=latent_dims).fit(self.X, self.Y, self.Z)
        wrap_mcca = MCCA(latent_dims=latent_dims).fit(self.X, self.Y, self.Z)
        wrap_kcca = KCCA(latent_dims=latent_dims).fit(self.X, self.Y, self.Z)
        corr_cca = wrap_cca.train_correlations[:, :, 0]
        corr_iter = wrap_iter.train_correlations[:, :, 0]
        corr_gcca = wrap_gcca.train_correlations[:, :, 0]
        corr_mcca = wrap_mcca.train_correlations[:, :, 0]
        corr_kcca = wrap_kcca.train_correlations[:, :, 0]
        # Check the score outputs are the right shape
        self.assertTrue(wrap_iter.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_gcca.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_mcca.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_kcca.score_list[0].shape == (self.X.shape[0], latent_dims))
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_iter, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_mcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_cca, corr_kcca, decimal=2))

    def test_regularized_methods(self):
        # Test that linear regularized methods match PLS solution when using maximum regularisation.
        latent_dims = 5
        c = 1
        wrap_kernel = KCCA(latent_dims=latent_dims, c=[c, c], kernel=['linear', 'linear']).fit(self.X,
                                                                                               self.Y)
        wrap_pls = PLS(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        wrap_mcca = MCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        wrap_rCCA = rCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        corr_gcca = wrap_gcca.train_correlations[0, 1]
        corr_mcca = wrap_mcca.train_correlations[0, 1]
        corr_kernel = wrap_kernel.train_correlations[0, 1]
        corr_pls = wrap_pls.train_correlations[0, 1]
        corr_rcca = wrap_rCCA.train_correlations[0, 1]
        # Check the correlations from each unregularized method are the same
        # self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_rcca, decimal=1))

    def test_non_negative_methods(self):
        latent_dims = 1
        wrap_nnelasticca = ElasticCCA(latent_dims=latent_dims, tol=1e-9, positive=True, l1_ratio=[0.5, 0.5],
                                      c=[1e-4, 1e-5]).fit(self.X, self.Y)
        wrap_als = CCA_ALS(latent_dims=latent_dims, tol=1e-9).fit(self.X, self.Y)
        wrap_nnals = CCA_ALS(latent_dims=latent_dims, tol=1e-9, positive=True).fit(self.X, self.Y)
        wrap_nnscca = SCCA(latent_dims=latent_dims, tol=1e-9, positive=True, c=[1e-4, 1e-5]).fit(self.X, self.Y)

    def test_sparse_methods(self):
        # Test sparsity inducing methods. At the moment just checks running.
        latent_dims = 5
        c1 = [1, 3]
        c2 = [1, 3]
        param_candidates = {'c': list(itertools.product(c1, c2))}
        wrap_pmd = PMD(latent_dims=latent_dims, random_state=self.rng).gridsearch_fit(self.X, self.Y,
                                                                                      param_candidates=param_candidates,
                                                                                      verbose=True, plot=True)
        c1 = [1e-4, 1e-5]
        c2 = [1e-4, 1e-5]
        param_candidates = {'c': list(itertools.product(c1, c2))}
        wrap_scca = SCCA(latent_dims=latent_dims, random_state=self.rng).gridsearch_fit(self.X, self.Y,
                                                                                        param_candidates=param_candidates,
                                                                                        verbose=True)
        wrap_elastic = ElasticCCA(latent_dims=latent_dims, random_state=self.rng).gridsearch_fit(self.X, self.Y,
                                                                                                 param_candidates=param_candidates,
                                                                                                 verbose=True)
        corr_pmd = wrap_pmd.train_correlations[0, 1]
        corr_scca = wrap_scca.train_correlations[0, 1]
        corr_elastic = wrap_elastic.train_correlations[0, 1]
        wrap_scca_admm = SCCA_ADMM(c=[1e-4, 1e-4]).fit(self.X, self.Y)
        wrap_scca = SCCA(c=[1e-4, 1e-4]).fit(self.X, self.Y)

    def test_weighted_GCCA_methods(self):
        # Test the 'fancy' additions to GCCA i.e. the view weighting and observation weighting.
        latent_dims = 5
        c = 0
        wrap_unweighted_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        wrap_deweighted_gcca = GCCA(latent_dims=latent_dims, c=[c, c], view_weights=[0.5, 0.5]).fit(
            self.X, self.Y)
        corr_unweighted_gcca = wrap_unweighted_gcca.train_correlations[0, 1]
        corr_deweighted_gcca = wrap_deweighted_gcca.train_correlations[0, 1]
        # Check the correlations from each unregularized method are the same
        K = np.ones((2, self.X.shape[0]))
        K[0, 200:] = 0
        wrap_unobserved_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y, K=K)
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_unweighted_gcca, corr_deweighted_gcca, decimal=1))

    def test_TCCA(self):
        # Tests tensor CCA methods
        latent_dims = 1
        wrap_tcca = TCCA(latent_dims=latent_dims, c=[0.2, 0.2]).fit(self.X, self.Y)
        wrap_ktcca = KTCCA(latent_dims=latent_dims, c=[0.2, 0.2]).fit(self.X, self.Y)
        corr_tcca = wrap_tcca.train_correlations[0, 1]
        corr_ktcca = wrap_ktcca.train_correlations[0, 1]
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_tcca, corr_ktcca, decimal=1))

    def test_cv_fit(self):
        # Test the CV method
        latent_dims = 5
        c1 = [0.1, 0.2]
        c2 = [0.1, 0.2]
        param_candidates = {'c': list(itertools.product(c1, c2))}
        wrap_unweighted_gcca = GCCA(latent_dims=latent_dims).gridsearch_fit(self.X, self.Y, folds=5,
                                                                            param_candidates=param_candidates,
                                                                            plot=True, jobs=3)
        wrap_deweighted_gcca = GCCA(latent_dims=latent_dims, view_weights=[0.5, 0.5]).gridsearch_fit(
            self.X, self.Y, folds=2, param_candidates=param_candidates)
        wrap_mcca = MCCA(latent_dims=latent_dims).gridsearch_fit(
            self.X, self.Y, folds=2, param_candidates=param_candidates)

    def test_l0(self):
        wrap_span_cca = SpanCCA(latent_dims=1, regularisation='l0', c=[2, 2]).fit(self.X, self.Y)
        wrap_swcca = SWCCA(latent_dims=1, c=[2, 2], sample_support=5).fit(self.X, self.Y)
        self.assertEqual((np.abs(wrap_span_cca.weights_list[0]) > 1e-5).sum(), 2)
        self.assertEqual((np.abs(wrap_span_cca.weights_list[1]) > 1e-5).sum(), 2)
        self.assertEqual((np.abs(wrap_swcca.weights_list[0]) > 1e-5).sum(), 2)
        self.assertEqual((np.abs(wrap_swcca.weights_list[1]) > 1e-5).sum(), 2)
        self.assertEqual((np.abs(wrap_swcca.loop.sample_weights) > 1e-5).sum(), 5)
        print()
