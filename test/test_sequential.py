import pytest
import numpy as np
from cca_zoo.linear import rCCA
from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.sequential import SequentialModel


@pytest.fixture
def data():
    # Generate some simulated data with 5 latent dimensions and 0.8 correlation
    data = LinearSimulatedData(view_features=[10, 10], latent_dims=5, correlation=0.8)
    X, Y = data.sample(200)
    return X, Y


@pytest.fixture
def estimator():
    # Create a rCCA estimator with a grid search for the regularization parameter
    rcca = rCCA()
    param_grid = {
        "c": [0.1, 0.2, 0.3],
    }
    gs = GridSearchCV(rcca, param_grid, cv=2, verbose=1, n_jobs=1)
    return gs


def test_fit(data, estimator):
    # Test the fit method of the SequentialModel class
    X, Y = data
    model = SequentialModel(
        estimator, latent_dimensions=10, permutation_test=True, p_threshold=0.05
    )
    model.fit([X, Y])
    # Check that the model has found at least one significant effect
    assert model.latent_dimensions > 0
    # Check that the model has weights for each view and each effect
    assert len(model.weights) == 2
    assert all(
        [weights.shape[1] == model.latent_dimensions for weights in model.weights]
    )
    # Check that the model has p-values for each effect
    assert len(model.p_values) == model.latent_dimensions
