import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from unittest import TestCase
import cca_zoo.wrappers


class TestWrapper(TestCase):

    def setUp(self):
        self.X = np.random.rand(20, 5)
        self.Y = np.random.rand(20, 5)
        self.Z = np.random.rand(20, 5)

    def tearDown(self):
        pass

    def test_unregularized_methods(self):
        latent_dims = 1
        wrap_als = cca_zoo.wrappers.CCA_ALS(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_mcca = cca_zoo.wrappers.MCCA(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_scikit = cca_zoo.wrappers.CCA_scikit(latent_dims=latent_dims).fit(self.X, self.Y)
        corr_als = wrap_als.train_correlations[0, 1]
        corr_gcca = wrap_gcca.train_correlations[0, 1]
        corr_mcca = wrap_mcca.train_correlations[0, 1]
        corr_scikit = wrap_scikit.train_correlations[0, 1]
        # Check the score outputs are the right shape
        self.assertTrue(wrap_als.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_gcca.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_mcca.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_scikit.score_list[0].shape == (self.X.shape[0], latent_dims))
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_als, corr_scikit, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_als, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_als, corr_mcca, decimal=2))

    def test_regularized_methods(self):
        # Test that linear regularized methods match PLS solution when using maximum regularisation
        latent_dims = 1
        params = {'c': [1, 1]}
        wrap_gcca = cca_zoo.wrappers.GCCA(latent_dims=latent_dims).fit(self.X, self.Y, params=params)
        wrap_mcca = cca_zoo.wrappers.MCCA(latent_dims=latent_dims).fit(self.X, self.Y, params=params)
        wrap_kernel = cca_zoo.wrappers.KCCA(latent_dims=latent_dims).fit(self.X, self.Y, params=params)
        wrap_pls = cca_zoo.wrappers.PLS_scikit(latent_dims=latent_dims).fit(self.X, self.Y)
        corr_gcca = wrap_gcca.train_correlations[0, 1]
        corr_mcca = wrap_mcca.train_correlations[0, 1]
        corr_kernel = wrap_kernel.train_correlations[0, 1]
        corr_pls = wrap_pls.train_correlations[0, 1]
        # Check the correlations from each unregularized method are the same
        #self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_gcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=2))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=2))

    def test_methods(self):
        pass

    #TODO
    def test_gridsearchfit(self):
        pass
