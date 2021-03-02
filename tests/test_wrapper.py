from unittest import TestCase

import numpy as np
import itertools
import cca_zoo.wrappers

np.random.seed(123)


class TestWrapper(TestCase):

    def setUp(self):
        self.X = np.random.rand(500, 30)
        self.Y = np.random.rand(500, 25)
        self.Z = np.random.rand(500, 20)

    def tearDown(self):
        pass

    def test_unregularized_methods(self):
        latent_dims = 5
        wrap_cca = cca_zoo.wrappers.CCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_iter = cca_zoo.wrappers.CCA_ALS(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_mcca = cca_zoo.wrappers.MCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_kcca = cca_zoo.wrappers.KCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        corr_cca = wrap_iter.train_correlations[0, 1]
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

    def test_regularized_methods(self):
        # Test that linear regularized methods match PLS solution when using maximum regularisation
        latent_dims = 5
        c = 1
        wrap_kernel = cca_zoo.wrappers.KCCA(latent_dims=latent_dims, c=[c, c], kernel='linear').fit(self.X, self.Y)
        wrap_pls = cca_zoo.wrappers.PLS(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        wrap_mcca = cca_zoo.wrappers.MCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        wrap_rCCA = cca_zoo.wrappers.rCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
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

    def test_sparse_methods(self):
        # Test that linear regularized methods match PLS solution when using maximum regularisation
        latent_dims = 5
        c1 = [1, 3]
        c2 = [1, 3]
        param_candidates = {'c': list(itertools.product(c1, c2))}
        wrap_pmd = cca_zoo.wrappers.PMD(latent_dims=latent_dims).gridsearch_fit(self.X, self.Y,
                                                                                param_candidates=param_candidates,
                                                                                verbose=True)
        c1 = [1e-3, 1e-4]
        c2 = [1e-3, 1e-4]
        param_candidates = {'c': list(itertools.product(c1, c2))}
        wrap_scca = cca_zoo.wrappers.SCCA(latent_dims=latent_dims).gridsearch_fit(self.X, self.Y,
                                                                                  param_candidates=param_candidates,
                                                                                  verbose=True)
        wrap_elastic = cca_zoo.wrappers.ElasticCCA(latent_dims=latent_dims).gridsearch_fit(self.X, self.Y,
                                                                                           param_candidates=param_candidates,
                                                                                           verbose=True)
        corr_pmd = wrap_pmd.train_correlations[0, 1]
        corr_scca = wrap_scca.train_correlations[0, 1]
        corr_elastic = wrap_elastic.train_correlations[0, 1]

    def test_weighted_GCCA_methods(self):
        # Test that linear regularized methods match PLS solution when using maximum regularisation
        latent_dims = 5
        c = 0
        wrap_unweighted_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims, c=[c, c]).fit(self.X, self.Y)
        wrap_deweighted_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims, c=[c, c], view_weights=[0.5, 0.5]).fit(
            self.X, self.Y)
        corr_unweighted_gcca = wrap_unweighted_gcca.train_correlations[0, 1]
        corr_deweighted_gcca = wrap_deweighted_gcca.train_correlations[0, 1]
        # Check the correlations from each unregularized method are the same
        K = np.ones((2, self.X.shape[0]))
        K[0, 200:] = 0
        wrap_unobserved_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims, c=[c, c], K=K).fit(self.X, self.Y)
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_unweighted_gcca, corr_deweighted_gcca, decimal=1))

    def test_methods(self):
        pass

    # TODO
    def test_gridsearchfit(self):
        pass
