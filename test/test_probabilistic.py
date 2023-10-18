import numpy as np
import pytest

from cca_zoo.data.simulated import JointDataGenerator
from cca_zoo.linear import CCA, PLS
from cca_zoo.probabilistic import ProbabilisticCCA
from cca_zoo.probabilistic._pls import ProbabilisticPLS
from cca_zoo.probabilistic._plsregression import ProbabilisticPLSRegression
from cca_zoo.probabilistic._rcca import ProbabilisticRCCA


@pytest.fixture
def setup_data():
    seed = 123
    latent_dims = 1
    data = JointDataGenerator(
        view_features=[5, 6],
        latent_dims=latent_dims,
        random_state=seed,
        structure='identity',
    )
    X, Y = data.sample(50)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    return X, Y, data.joint_cov


def test_cca_vs_probabilisticCCA(setup_data):
    X, Y, joint = setup_data
    # Models and fit
    cca = CCA(latent_dimensions=1)
    pcca = ProbabilisticCCA(latent_dimensions=1, random_state=10)
    cca.fit([X, Y])
    pcca.fit([X, Y])

    # Assert: Calculate correlation coefficient and ensure it's greater than 0.95
    z = cca.transform([X, Y])[0]
    z_p = np.array(pcca.transform([X, None]))
    # correlation between cca and pcca
    correlation_matrix = np.abs(np.corrcoef(z.reshape(-1), z_p.reshape(-1)))
    correlation = correlation_matrix[0, 1]

    assert (
        correlation > 0.9
    ), f"Expected correlation greater than 0.95, got {correlation}"


def test_cca_vs_probabilisticPLS(setup_data):
    X, Y, joint = setup_data
    # Models and fit
    cca = CCA(latent_dimensions=1)
    pls = PLS(latent_dimensions=1)
    ppls = ProbabilisticPLS(latent_dimensions=1, random_state=1, learning_rate=1e-4, n_iter=20000)

    cca.fit([X, Y])
    pls.fit([X, Y])
    ppls.fit([X, Y])
    model_joint = ppls.joint()

    # Assert: Calculate correlation coefficient and ensure it's greater than 0.98
    z_cca = cca.transform([X, Y])[0]
    z_pls = pls.transform([X, Y])[0]
    z_p, z_pstd = np.array(ppls.transform([X, None], return_std=True))
    # correlation between pls and ppls
    correlation_matrix = np.abs(np.corrcoef(z_pls.reshape(-1), z_p.reshape(-1)))
    correlation_pls = correlation_matrix[0, 1]

    correlation_matrix = np.abs(np.corrcoef(z_cca.reshape(-1), z_p.reshape(-1)))
    correlation_cca = correlation_matrix[0, 1]

    assert (
        correlation_pls > correlation_cca
    ), f"Expected correlation with PLS greater than CCA, got {correlation_pls} and {correlation_cca}"


def test_cca_vs_probabilisticRidgeCCA(setup_data):
    X, Y, joint = setup_data
    # Initialize models with different regularization parameters
    prcca_pls = ProbabilisticRCCA(latent_dimensions=1, random_state=10, c=0.999)
    prcca_cca = ProbabilisticRCCA(latent_dimensions=1, random_state=10, c=0)
    # Fit and Transform using ProbabilisticRCCA with large and small regularization
    prcca_cca.fit([X, Y])
    prcca_pls.fit([X, Y])

    z_ridge_cca = np.array(prcca_cca.transform([X, None]))
    z_ridge_pls = np.array(prcca_pls.transform([X, None]))

    # Fit and Transform using classical CCALoss and PLS
    cca = CCA(latent_dimensions=1)
    pls = PLS(latent_dimensions=1)

    cca.fit([X, Y])
    pls.fit([X, Y])

    z_cca = np.array(cca.transform([X, Y])[0])
    z_pls = np.array(pls.transform([X, Y])[0])

    # Assert: Correlations should be high when ProbabilisticRCCA approximates CCALoss and PLS
    corr_matrix_cca = np.abs(np.corrcoef(z_cca.reshape(-1), z_ridge_cca.reshape(-1)))
    corr_cca = corr_matrix_cca[0, 1]
    assert corr_cca > 0.9, f"Expected correlation greater than 0.9, got {corr_cca}"

    corr_matrix_pls = np.abs(np.corrcoef(z_pls.reshape(-1), z_ridge_pls.reshape(-1)))
    corr_pls = corr_matrix_pls[0, 1]
    assert corr_pls > 0.9, f"Expected correlation greater than 0.9, got {corr_pls}"
