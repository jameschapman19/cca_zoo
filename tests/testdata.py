from unittest import TestCase

from sklearn.utils.validation import check_random_state

from cca_zoo.data import generate_covariance_data
from cca_zoo.models import MCCA, CCA


class TestData(TestCase):

    def setUp(self):
        self.rng = check_random_state(0)
        self.X = self.rng.rand(500, 20)
        self.Y = self.rng.rand(500, 21)
        self.Z = self.rng.rand(500, 22)

    def tearDown(self):
        pass

    def test_data_gen(self):
        (x, y, z), true_feats = generate_covariance_data(1000, [10, 11, 12], 1, [0.5, 0.5, 0.5], correlation=0.5,
                                                         structure=['identity', 'identity', 'identity'])
        cca = CCA().fit(x[:500], y[:500])
        cca_pred = cca.predict_corr(x[500:], y[500:])
        mcca = MCCA().fit(x[:500], y[:500], z[:500])
        mcca_pred = mcca.predict_corr(x[500:], y[500:], z[500:])

        (x, y), true_feats = generate_covariance_data(1000, [10, 11], 1, [0.5, 0.5], correlation=0.5,
                                                      structure=['gaussian', 'toeplitz'])
