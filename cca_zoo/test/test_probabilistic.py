import numpy as np
import pytest

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.models import CCA


def test_PCCA():
    # some might not have access to jax/numpyro so leave this as an optional test locally.
    numpyro = pytest.importorskip("numpyro")
    from cca_zoo.probabilisticmodels import ProbabilisticCCA

    np.random.seed(0)
    # Tests tensor CCA methods
    X, Y = LinearSimulatedData([5, 5]).sample(100)
    latent_dims = 1
    cca = CCA(latent_dims=latent_dims).fit([X, Y])
    pcca = ProbabilisticCCA(
        latent_dims=latent_dims, num_warmup=1000, num_samples=1000
    ).fit([X, Y])
    # Test that vanilla CCA and VCCA produce roughly similar latent space ie they are correlated
    assert (
        np.abs(
            np.corrcoef(
                cca.transform([X, Y])[1].T,
                pcca.posterior_samples["z"].mean(axis=0)[:, 0],
            )[0, 1]
        )
        > 0.9
    )
