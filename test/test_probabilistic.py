import numpy as np
import pytest

from cca_zoo.datasets import LatentVariableData
from cca_zoo.linear import CCA
from cca_zoo.probabilistic import ProbabilisticCCA


@pytest.fixture
def setup_data():
    seed = 123
    latent_dims = 1
    data = LatentVariableData(
        view_features=[5, 6],
        latent_dims=latent_dims,
        random_state=seed,
        structure="identity",
    )
    X, Y = data.sample(500)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    return X, Y, data


# def test_cca_vs_probabilisticCCA(setup_data):
#     X, Y, data = setup_data
#     # Models and fit
#     cca = CCA(latent_dimensions=1)
#     pcca = ProbabilisticCCA(latent_dimensions=1, random_state=10)
#     cca.fit([X, Y])
#     pcca.fit([X, Y])
#
#     # Assert: Calculate correlation coefficient and ensure it's greater than 0.95
#     z = cca.transform([X, Y])[0]
#     z_p = np.array(pcca.transform([X, Y]))
#     # correlation between cca and pcca
#     correlation_matrix = np.abs(np.corrcoef(z.reshape(-1), z_p.reshape(-1)))
#     correlation = correlation_matrix[0, 1]
#
#     assert (
#         correlation > 0.9
#     ), f"Expected correlation greater than 0.95, got {correlation}"
