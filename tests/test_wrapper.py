from unittest import TestCase

import numpy as np

import cca_zoo.wrappers

np.random.seed(123)


class TestWrapper(TestCase):

    def setUp(self):
        self.X = np.random.rand(50, 40000)
        self.Y = np.random.rand(50, 40000)
        self.Z = np.random.rand(50, 4000)

    def tearDown(self):
        pass

    def test_unregularized_methods(self):
        latent_dims = 5
        wrap_iter = cca_zoo.wrappers.CCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_mcca = cca_zoo.wrappers.MCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_kcca = cca_zoo.wrappers.KCCA(latent_dims=latent_dims).fit(self.X, self.Y)
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
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_mcca, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_gcca, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_iter, corr_kcca, decimal=1))

    def test_regularized_methods(self):
        # Test that linear regularized methods match PLS solution when using maximum regularisation
        latent_dims = 5
        c = 0.5
        wrap_kernel = cca_zoo.wrappers.KCCA(latent_dims=latent_dims).fit(self.X, self.Y, c=[c, c], kernel='linear')
        wrap_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims).fit(self.X, self.Y, c=[c, c])
        wrap_mcca = cca_zoo.wrappers.MCCA(latent_dims=latent_dims).fit(self.X, self.Y, c=[c, c])
        wrap_pls = cca_zoo.wrappers.PLS(latent_dims=latent_dims).fit(self.X, self.Y)
        corr_gcca = wrap_gcca.train_correlations[0, 1]
        corr_mcca = wrap_mcca.train_correlations[0, 1]
        corr_kernel = wrap_kernel.train_correlations[0, 1]
        corr_pls = wrap_pls.train_correlations[0, 1]
        # Check the correlations from each unregularized method are the same
        # self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=1))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=1))

    def test_methods(self):
        pass

    # TODO
    def test_gridsearchfit(self):
        pass
