import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from unittest import TestCase
import cca_zoo.innerloop


class TestAlsInnerLoop(TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 10)
        self.Y = np.random.rand(10, 10)
        self.Z = np.random.rand(10, 10)

    def tearDown(self):
        pass

    def test_iterate(self):
        pls = cca_zoo.innerloop.InnerLoop(self.X, self.Y)
        generalized_pls = cca_zoo.innerloop.InnerLoop(self.X, self.Y, self.Z)
        als = cca_zoo.innerloop.CCAInnerLoop(self.X, self.Y)
        generalized_als = cca_zoo.innerloop.CCAInnerLoop(self.X, self.Y, self.Z)
        self.assertIsNone(
            np.testing.assert_almost_equal(np.linalg.norm(pls.weights[0]), np.linalg.norm(pls.weights[1]), 1))
        self.assertIsNone(
            np.testing.assert_almost_equal(np.linalg.norm(als.scores[0]), np.linalg.norm(als.scores[1]), 1))
        self.assertIsNone(
            np.testing.assert_almost_equal(np.linalg.norm(generalized_pls.weights[0]),
                                           np.linalg.norm(generalized_pls.weights[1]),
                                           np.linalg.norm(generalized_pls.weights[2]), 1))
        self.assertIsNone(
            np.testing.assert_almost_equal(np.linalg.norm(generalized_als.scores[0]),
                                           np.linalg.norm(generalized_als.scores[1]),
                                           np.linalg.norm(generalized_als.scores[2]), 1))

    def test_regularized(self):
        scca = cca_zoo.innerloop.SCCAInnerLoop(self.X, self.Y, c=[0.0001, 0.0001])
        scca_gen = cca_zoo.innerloop.SCCAInnerLoop(self.X, self.Y, c=[0.0001, 0.0001], generalized=True)
        park = cca_zoo.innerloop.ParkhomenkoInnerLoop(self.X, self.Y,c=[0.0001, 0.0001])
        park_gen = cca_zoo.innerloop.ParkhomenkoInnerLoop(self.X, self.Y,c=[0.0001, 0.0001], generalized=True)
        params = {'c': [2, 2]}
        pmd = cca_zoo.innerloop.PMDInnerLoop(self.X, self.Y, c=[2,2])
        pmd_gen = cca_zoo.innerloop.PMDInnerLoop(self.X, self.Y, c=[2,2], generalized=True)
