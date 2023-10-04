import numpy as np
import pytest

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.linear import CCA


@pytest.mark.parametrize(
    "view_features, latent_dims, correlation, atol",
    [
        ([10, 10], 5, [0.9, 0.8, 0.7, 0.6, 0.5], 0.1),
        # You can add more parameter sets to test different scenarios
    ],
)
def test_cca_on_simulated_data_maintains_expected_correlation(
    view_features, latent_dims, correlation, atol
):
    # Generate Data
    data = LinearSimulatedData(
        view_features=view_features, latent_dims=latent_dims, correlation=correlation
    )
    x_train, y_train = data.sample(1000)
    x_test, y_test = data.sample(1000)

    # Ensure train and test data are different
    assert not np.array_equal(x_train, x_test)
    assert not np.array_equal(y_train, y_test)

    # Train model
    model = CCA(latent_dimensions=latent_dims)
    model.fit((x_train, y_train))

    # Test model
    assert np.allclose(model.average_pairwise_correlations((x_test, y_test)), np.array(correlation), atol=atol)


# Additional test to verify the shape of generated data
def test_simulated_data_shapes():
    data = LinearSimulatedData(
        view_features=[10, 12], latent_dims=4, correlation=[0.8, 0.7, 0.6, 0.5]
    )
    x_train, y_train = data.sample(500)
    assert x_train.shape == (500, 10)
    assert y_train.shape == (500, 12)
