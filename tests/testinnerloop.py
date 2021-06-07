from unittest import TestCase

import numpy as np

import cca_zoo.models.innerloop


class TestInnerLoop(TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 10)
        self.Y = np.random.rand(10, 10)
        self.Z = np.random.rand(10, 10)

    def tearDown(self):
        pass

    def test_regularized(self):
        park = cca_zoo.models.innerloop.ParkhomenkoInnerLoop(c=[0.0001, 0.0001]).fit(self.X, self.Y)
        park_gen = cca_zoo.models.innerloop.ParkhomenkoInnerLoop(c=[0.0001, 0.0001], generalized=True).fit(self.X,
                                                                                                           self.Y)
        params = {'c': [2, 2]}
        pmd = cca_zoo.models.innerloop.PMDInnerLoop(c=[2, 2]).fit(self.X, self.Y)
        pmd_gen = cca_zoo.models.innerloop.PMDInnerLoop(c=[2, 2], generalized=True).fit(self.X, self.Y)
