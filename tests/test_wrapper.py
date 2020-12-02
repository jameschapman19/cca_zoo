import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from unittest import TestCase
import cca_zoo.wrapper


class TestWrapper(TestCase):

    def setUp(self):
        self.X = np.random.rand(20, 5)
        self.Y = np.random.rand(20, 5)
        self.Z = np.random.rand(20, 5)

    def tearDown(self):
        pass

    def test_unregularized_methods(self):
        latent_dims = 1
        wrap_als = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims).fit(self.X, self.Y)
        wrap_gep = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='gep').fit(self.X, self.Y)
        wrap_gcca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='gcca').fit(self.X, self.Y)
        wrap_mcca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='mcca').fit(self.X, self.Y)
        wrap_scikit = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='scikit').fit(self.X, self.Y)
        corr_als = wrap_als.train_correlations[0, 1]
        corr_gep = wrap_gep.train_correlations[0, 1]
        corr_gcca = wrap_gcca.train_correlations[0, 1]
        corr_mcca = wrap_mcca.train_correlations[0, 1]
        corr_scikit = wrap_scikit.train_correlations[0, 1]
        # Check the score outputs are the right shape
        self.assertTrue(wrap_als.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_als.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_als.score_list[0].shape == (self.X.shape[0], latent_dims))
        self.assertTrue(wrap_als.score_list[0].shape == (self.X.shape[0], latent_dims))
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_als, corr_scikit, decimal=4))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_als, corr_gep, decimal=4))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_als, corr_gcca, decimal=4))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_als, corr_mcca, decimal=4))

    def test_regularized_methods(self):
        # Test that linear regularized methods match PLS solution when using maximum regularisation
        latent_dims = 1
        params = {'c': [1, 1]}
        wrap_gep = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='gep').fit(self.X, self.Y, params=params)
        wrap_mcca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='mcca').fit(self.X, self.Y, params=params)
        wrap_kernel = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='kernel').fit(self.X, self.Y,
                                                                                            params=params)
        wrap_pls = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='pls').fit(self.X, self.Y)
        corr_gep = wrap_gep.train_correlations[0, 1]
        corr_mcca = wrap_mcca.train_correlations[0, 1]
        corr_kernel = wrap_kernel.train_correlations[0, 1]
        corr_pls = wrap_pls.train_correlations[0, 1]
        # Check the correlations from each unregularized method are the same
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_gep, decimal=3))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=3))
        self.assertIsNone(np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=3))

    def test_methods(self):
        pass
