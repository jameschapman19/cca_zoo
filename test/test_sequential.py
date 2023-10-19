import pytest
import numpy as np

from cca_zoo.datasets import JointData
from cca_zoo.linear import rCCA
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.sequential import SequentialModel


# Fixtures
@pytest.fixture
def simulated_data():
    data_generator = JointData(view_features=[4, 6], latent_dims=5, correlation=0.8)
    X, Y = data_generator.sample(50)
    return X, Y


@pytest.fixture
def grid_search_estimator():
    rcca = rCCA()
    parameters = {
        "c": [0.1, 0.2, 0.3],
    }
    grid_search = GridSearchCV(rcca, parameters, cv=2, verbose=1, n_jobs=1)
    return grid_search


# Test
def test_sequential_model_fits_and_identifies_effects(
    simulated_data, grid_search_estimator
):
    X, Y = simulated_data
    sequential_model = SequentialModel(
        grid_search_estimator,
        latent_dimensions=10,
        permutation_test=True,
        p_threshold=0.05,
    )

    # Training
    sequential_model.fit([X, Y])

    # Test if the model identified significant effects
    assert sequential_model.latent_dimensions > 0

    # Test if the model has correct weight attributes
    assert len(sequential_model.weights) == 2
    assert all(
        [
            weight.shape[1] == sequential_model.latent_dimensions
            for weight in sequential_model.weights
        ]
    )

    # Test if model has correct number of p-values
    assert len(sequential_model.p_values) == sequential_model.latent_dimensions
