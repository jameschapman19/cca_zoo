import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.utils import check_random_state

from cca_zoo.models import (
    CCA,
    PLS,
)
from cca_zoo.models._iterative._svd import CCASVD

n = 50
rng = check_random_state(0)
X = rng.rand(n, 10)
Y = rng.rand(n, 11)
Z = rng.rand(n, 12)
X_sp = sp.random(n, 10, density=0.5, random_state=rng)
Y_sp = sp.random(n, 11, density=0.5, random_state=rng)
# centre the data
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)
Z -= Z.mean(axis=0)
X_sp -= X_sp.mean(axis=0)
Y_sp -= Y_sp.mean(axis=0)


def test_stochastic_pls():
    pytest.importorskip("torch")
    from cca_zoo.models import PLSEY, PLSGHA, PLSStochasticPower
    from torch import manual_seed

    manual_seed(42)
    plsey = PLSEY(
        latent_dims=3,
        epochs=1000,
        batch_size=None,
        random_state=1,
        learning_rate=1e-2,
        convergence_checking=False,
        tol=1e-7,
    ).fit((X, Y))
    plsgh = PLSGHA(
        latent_dims=3,
        epochs=1000,
        batch_size=None,
        random_state=1,
        learning_rate=1e-2,
    ).fit((X, Y))
    pls = PLS(latent_dims=3).fit((X, Y))
    spls = PLSStochasticPower(
        latent_dims=3, epochs=1000, batch_size=None, learning_rate=1e-1, random_state=1
    ).fit((X, Y))
    pls_score = pls.score((X, Y))
    spls_score = np.abs(spls.score((X, Y)))
    plsey_score = plsey.score((X, Y))
    ghapls_score = plsgh.score((X, Y))
    # check all methods are similar to pls
    assert np.allclose(pls_score, spls_score, atol=1e-1)
    assert np.allclose(pls_score.sum(), plsey_score.sum(), atol=1e-1)
    assert np.allclose(pls_score.sum(), ghapls_score.sum(), atol=1e-1)


def test_stochastic_cca():
    pytest.importorskip("torch")
    from cca_zoo.models import CCAEY, CCAGH

    cca = CCA(latent_dims=3).fit((X, Y))
    ccaey = CCAEY(
        latent_dims=3,
        epochs=1000,
        batch_size=25,
        random_state=1,
        learning_rate=1e-2,
    ).fit((X, Y))
    ccagh = CCAGH(
        latent_dims=3,
        epochs=1000,
        batch_size=50,
        random_state=1,
        learning_rate=1e-2,
    ).fit((X, Y))
    ccasvd = CCASVD(
        latent_dims=3,
        epochs=1000,
        batch_size=25,
        random_state=1,
        learning_rate=1e-2,
    ).fit((X, Y))
    cca_score = cca.score((X, Y))
    ccaey_score = ccaey.score((X, Y))
    ccagh_score = ccagh.score((X, Y))
    ccasvd_score = ccasvd.score((X, Y))
    # check all methods are similar to cca
    assert np.allclose(cca_score.sum(), ccaey_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccagh_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccasvd_score.sum(), atol=1e-1)
    print()
