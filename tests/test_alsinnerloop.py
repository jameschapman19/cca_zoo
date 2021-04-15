from unittest import TestCase

import numpy as np

import cca_zoo.innerloop


class TestAlsInnerLoop(TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 10)
        self.Y = np.random.rand(10, 10)
        self.Z = np.random.rand(10, 10)

    def tearDown(self):
        pass

    def test_iterate(self):
        pls = cca_zoo.innerloop._InnerLoop().fit(self.X, self.Y)
        generalized_pls = cca_zoo.innerloop._InnerLoop().fit(self.X, self.Y, self.Z)
        als = cca_zoo.innerloop.CCAInnerLoop().fit(self.X, self.Y)
        generalized_als = cca_zoo.innerloop.CCAInnerLoop().fit(self.X, self.Y, self.Z)
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
        scca = cca_zoo.innerloop.SCCAInnerLoop(c=[0.0001, 0.0001]).fit(self.X, self.Y)
        scca_gen = cca_zoo.innerloop.SCCAInnerLoop(c=[0.0001, 0.0001], generalized=True).fit(self.X, self.Y)
        park = cca_zoo.innerloop.ParkhomenkoInnerLoop(c=[0.0001, 0.0001]).fit(self.X, self.Y)
        park_gen = cca_zoo.innerloop.ParkhomenkoInnerLoop(c=[0.0001, 0.0001], generalized=True).fit(self.X, self.Y)
        params = {'c': [2, 2]}
        pmd = cca_zoo.innerloop.PMDInnerLoop(c=[2, 2]).fit(self.X, self.Y)
        pmd_gen = cca_zoo.innerloop.PMDInnerLoop(c=[2, 2], generalized=True).fit(self.X, self.Y)
