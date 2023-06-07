import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.utils import check_random_state

from cca_zoo.models import CCA, PLS, MCCA

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

latent_dims = 3
epochs = 250
batch_size = 10
learning_rate = 1e-1
random_state = 1


def scale_objective(obj):
    # log scale the objective if negative, otherwise nan
    obj = np.array(obj)
    return np.sign(obj) * np.log(np.abs(obj) + 1e-10)


def test_stochastic_pls():
    pytest.importorskip("torch")
    from torch import manual_seed

    from cca_zoo.models import PLSEY, PLSSVD, PLSStochasticPower

    pls = PLS(latent_dims=3).fit((X, Y))
    manual_seed(42)
    plsey = PLSEY(
        latent_dims=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        track=True,
        verbose=False,
    ).fit((X, Y))
    plssvd = PLSSVD(
        latent_dims=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        track=True,
        verbose=False,
    ).fit((X, Y))
    spls = PLSStochasticPower(
        latent_dims=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        track=True,
        verbose=False,
    ).fit((X, Y))

    def scale_transform(model, X, Y):
        Zx, Zy = model.transform((X, Y))
        Zx /= np.linalg.norm(model.weights[0], axis=0, keepdims=True)
        Zy /= np.linalg.norm(model.weights[1], axis=0, keepdims=True)
        return np.abs(np.cov(Zx, Zy, rowvar=False)[:latent_dims, latent_dims:])

    pls_score = scale_transform(pls, X, Y)
    spls_score = scale_transform(spls, X, Y)
    plsey_score = scale_transform(plsey, X, Y)
    plssvd_score = scale_transform(plssvd, X, Y)
    # check all methods are similar to pls
    assert np.allclose(np.trace(pls_score), np.trace(spls_score), atol=1e-1)
    assert np.allclose(np.trace(pls_score), np.trace(plsey_score), atol=1e-1)
    assert np.allclose(np.trace(pls_score), np.trace(plssvd_score), atol=1e-1)


def test_stochastic_cca():
    pytest.importorskip("torch")
    from cca_zoo.models import CCAEY, CCAGH, CCASVD

    cca = CCA(latent_dims=3).fit((X, Y))
    ccaey = CCAEY(
        latent_dims=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        track=True,
        verbose=False,
    ).fit((X, Y))
    ccagh = CCAGH(
        latent_dims=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        track=True,
        verbose=False,
    ).fit((X, Y))
    ccasvd = CCASVD(
        latent_dims=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        track=True,
        verbose=False,
    ).fit((X, Y))
    cca_score = cca.score((X, Y))
    ccaey_score = ccaey.score((X, Y))
    ccagh_score = ccagh.score((X, Y))
    ccasvd_score = ccasvd.score((X, Y))
    # check all methods are similar to cca
    assert np.allclose(cca_score.sum(), ccaey_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccagh_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccasvd_score.sum(), atol=1e-1)
