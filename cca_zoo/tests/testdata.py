from unittest import TestCase

import numpy as np
from sklearn.utils.validation import check_random_state

from cca_zoo.data import Noisy_MNIST_Dataset, Tangled_MNIST_Dataset, Split_MNIST_Dataset
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

    def test_data_rand(self):
        (x, y), true_feats = generate_covariance_data(1000, [10, 11], 1, [0.5, 0.5], correlation=0.5,
                                                      structure='random')

    def test_data_gen(self):
        (x, y, z), true_feats = generate_covariance_data(1000, [10, 11, 12], 1, [0.5, 0.5, 0.5], correlation=0.5,
                                                         structure=['identity', 'identity', 'identity'])
        cca = CCA().fit(x[:500], y[:500])
        cca_pred = cca.score(x[500:], y[500:])
        mcca = MCCA().fit(x[:500], y[:500], z[:500])
        mcca_pred = mcca.score(x[500:], y[500:], z[500:])

        (x, y), true_feats = generate_covariance_data(1000, [10, 11], 1, [0.5, 0.5], correlation=0.5,
                                                      structure=['gaussian', 'toeplitz'])

    def test_deep_data(self):
        dataset = Noisy_MNIST_Dataset(mnist_type='FashionMNIST', train=True)
        (train_view_1, train_view_2), (train_rotations, train_labels) = dataset.to_numpy(np.arange(10))
        dataset = Tangled_MNIST_Dataset(mnist_type='FashionMNIST', train=True)
        (train_view_1, train_view_2), (train_rotations_1, train_rotations_2, train_labels) = dataset.to_numpy(
            np.arange(10))
        dataset = Split_MNIST_Dataset(mnist_type='FashionMNIST', train=True)
        (train_view_1, train_view_2), (train_labels) = dataset.to_numpy(np.arange(10))
