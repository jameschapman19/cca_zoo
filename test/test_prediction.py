import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.utils.validation import check_random_state

from cca_zoo.linear import MPLS


# Setup a fixture for common data
@pytest.fixture
def data():
    n = 50
    rng = check_random_state(0)
    X = rng.rand(n, 11)
    Y = rng.rand(n, 10)
    Z = rng.rand(n, 12)
    X_sp = sp.random(n, 10, density=0.5, random_state=rng)
    Y_sp = sp.random(n, 11, density=0.5, random_state=rng)
    # centre the data
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    Z -= Z.mean(axis=0)
    X_sp -= X_sp.mean(axis=0)
    Y_sp -= Y_sp.mean(axis=0)
    return X, Y, Z, X_sp, Y_sp


def test_prediction(data):
    X, Y, Z, _, _ = data
    latent_dims = 10
    model = MPLS(latent_dimensions=latent_dims)
    model.fit([Y, Y])
    _, Y_ = model.predict([Y, None])
    # Check for perfect reconstruction
    assert np.testing.assert_array_almost_equal(Y, Y_, decimal=1) is None
