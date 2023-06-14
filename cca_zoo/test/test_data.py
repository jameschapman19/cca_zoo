"""
Test whether linear simulated data has the correct correlations when we run cca on it
"""
from cca_zoo.data.simulated import LinearSimulatedData
import numpy as np


def test_data_correlation():
    data = LinearSimulatedData(
        view_features=[10, 10], latent_dims=5, correlation=[0.9, 0.8, 0.7, 0.6, 0.5]
    )
    x_train, y_train = data.sample(1000)
    x_test, y_test = data.sample(1000)
    from cca_zoo.classical import CCA

    model = CCA(latent_dimensions=5)
    model.fit((x_train, y_train))
    assert np.allclose(
        model.score((x_test, y_test)), np.array([0.9, 0.8, 0.7, 0.6, 0.5]), atol=0.1
    )
