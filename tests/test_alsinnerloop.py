import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from unittest import TestCase
import cca_zoo.alsinnerloop


class TestAlsInnerLoop(TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 10)
        self.Y = np.random.rand(10, 10)
        self.Z = np.random.rand(10, 10)

    def tearDown(self):
        pass

    def test_iterate(self):
        als = cca_zoo.alsinnerloop.AlsInnerLoop(self.X, self.Y)
        generalized_als = cca_zoo.alsinnerloop.AlsInnerLoop(self.X, self.Y, self.Z)
        self.assertIsNone(
            np.testing.assert_almost_equal(np.linalg.norm(als.scores[0]), np.linalg.norm(als.scores[1]), 1))
        self.assertIsNone(
            np.testing.assert_almost_equal(np.linalg.norm(generalized_als.scores[0]),
                                           np.linalg.norm(generalized_als.scores[1]),
                                           np.linalg.norm(generalized_als.scores[2]), 1))

    def test_regularized(self):

        params = {'c': [0.0001, 0.0001]}
        scca = cca_zoo.alsinnerloop.AlsInnerLoop(self.X, self.Y, method='scca', params=params)
        scca_gen = cca_zoo.alsinnerloop.AlsInnerLoop(self.X, self.Y, method='scca', params=params, generalized=True)
        park = cca_zoo.alsinnerloop.AlsInnerLoop(self.X, self.Y, method='parkhomenko', params=params)
        park_gen = cca_zoo.alsinnerloop.AlsInnerLoop(self.X, self.Y, method='parkhomenko', params=params,
                                                     generalized=True)
        params = {'c': [2, 2]}
        pmd = cca_zoo.alsinnerloop.AlsInnerLoop(self.X, self.Y, method='pmd', params=params)
        pmd_gen = cca_zoo.alsinnerloop.AlsInnerLoop(self.X, self.Y, method='pmd', params=params, generalized=True)
