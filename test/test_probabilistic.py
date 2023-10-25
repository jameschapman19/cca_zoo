import numpy as np
import pytest

from cca_zoo.datasets import LatentVariableData
from cca_zoo.linear import CCA, PLS
from cca_zoo.probabilistic import ProbabilisticCCA
from cca_zoo.probabilistic._pls import ProbabilisticPLS


@pytest.fixture
def setup_data():
    seed = 123
    latent_dims = 1
    data = LatentVariableData(
        view_features=[3, 3],
        latent_dims=latent_dims,
        random_state=seed,
        structure="identity",
    )
    X, Y = data.sample(20)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    return X, Y, data


def test_cca_vs_probabilisticCCA(setup_data):
    X, Y, data = setup_data
    # Models and fit
    cca = CCA(latent_dimensions=1)
    pcca = ProbabilisticCCA(latent_dimensions=1, random_state=10)
    cca.fit([X, Y])
    pcca.fit([X, Y])

    # Assert: Calculate correlation coefficient and ensure it's greater than 0.95
    z = cca.transform([X, Y])[0]
    # correlation between cca and pcca
    correlation_matrix = np.abs(
        np.corrcoef(z.reshape(-1), pcca.params["z_loc"].reshape(-1))
    )
    correlation = correlation_matrix[0, 1]

    assert correlation > 0.8


def test_pls_vs_probabilisticPLS(setup_data):
    X, Y, data = setup_data
    # Models and fit
    pls = PLS(latent_dimensions=1)
    ppls = ProbabilisticPLS(latent_dimensions=1, random_state=10)
    pls.fit([X, Y])
    ppls.fit([X, Y])

    # Assert: Calculate correlation coefficient and ensure it's greater than 0.95
    z = pls.transform([X, Y])[0]
    # correlation between cca and pcca
    correlation_matrix = np.abs(
        np.corrcoef(z.reshape(-1), ppls.params["z_loc"].reshape(-1))
    )
    correlation = correlation_matrix[0, 1]

    assert correlation > 0.8
