import numpy as np
import pytest

from cca_zoo.datasets import JointData
from cca_zoo.linear import CCA


@pytest.mark.parametrize(
    "view_features, latent_dimensions, correlation, atol",
    [
        ([10, 10], 5, [0.9, 0.8, 0.7, 0.6, 0.5], 0.1),
        # You can add more parameter sets to test different scenarios
    ],
)
def test_cca_on_simulated_data_maintains_expected_correlation(
    view_features, latent_dimensions, correlation, atol
):
    # Generate Data
    data = JointData(
        view_features=view_features, latent_dimensions=latent_dimensions, correlation=correlation
    )
    x_train, y_train = data.sample(1000)
    x_test, y_test = data.sample(1000)

    # Ensure train and test data are different
    assert not np.array_equal(x_train, x_test)
    assert not np.array_equal(y_train, y_test)

    # Train model
    model = CCA(latent_dimensions=latent_dimensions)
    model.fit((x_train, y_train))

    # Test model
    assert np.allclose(
        model.average_pairwise_correlations((x_test, y_test)),
        np.array(correlation),
        atol=atol,
    )


# Additional test to verify the shape of generated data
def test_simulated_data_shapes():
    data = JointData(
        view_features=[4, 6], latent_dimensions=4, correlation=[0.8, 0.7, 0.6, 0.5]
    )
    x_train, y_train = data.sample(5)
    assert x_train.shape == (5, 4)
    assert y_train.shape == (5, 6)
