import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from unittest import TestCase
import cca_zoo.deepwrapper

class TestDeepWrapper(TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 10)
        self.Y = np.random.rand(10, 10)
        self.Z = np.random.rand(10, 10)
