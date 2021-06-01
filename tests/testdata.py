from unittest import TestCase

import numpy as np

from cca_zoo.data import generate_covariance_data
from cca_zoo.models import MCCA, CCA

np.random.seed(123)


class TestData(TestCase):

    def setUp(self):
        self.X = np.random.rand(500, 20)
        self.Y = np.random.rand(500, 21)
        self.Z = np.random.rand(500, 22)

    def tearDown(self):
        pass

    def test_data_gen(self):
        (x, y, z), true_feats = generate_covariance_data(1000, [10, 11, 12], 1, [0.5, 0.5, 0.5], correlation=0.5)
        cca = CCA().fit(x[:500], y[:500])
        cca_pred = cca.predict_corr(x[500:], y[500:])
        mcca = MCCA().fit(x[:500], y[:500], z[:500])
        mcca_pred = mcca.predict_corr(x[500:], y[500:], z[500:])
