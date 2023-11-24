import numpy as np
from sklearn.utils import check_random_state
from torch import manual_seed

manual_seed(42)
from cca_zoo.linear import (
    CCA,
    PLS,
    CCA_SVD,
    CCA_GHA,
    CCA_EY,
    PLS_EY,
    PLSStochasticPower,
)

n = 100
rng = check_random_state(0)
X = rng.rand(n, 4)
Y = rng.rand(n, 6)
Z = rng.rand(n, 8)
# centre the data
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)
Z -= Z.mean(axis=0)
X /= X.std(axis=0)
Y /= Y.std(axis=0)
Z /= Z.std(axis=0)

latent_dims = 2
epochs = 20
batch_size = 4
random_state = 1
trainer_kwargs = dict(
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)


def scale_transform(model, X, Y):
    model.weights_ = [
        w / np.linalg.norm(w, axis=0, keepdims=True) for w in model.weights_
    ]
    Zx, Zy = model.transform((X, Y))
    return np.abs(np.cov(Zx, Zy, rowvar=False)[:latent_dims, latent_dims:])


def test_batch_pls():
    epochs = 50
    pls = PLS(latent_dimensions=latent_dims).fit((X, Y))
    plsey = PLS_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    spls = PLSStochasticPower(
        latent_dimensions=latent_dims,
        epochs=epochs,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    pls_score = scale_transform(pls, X, Y)
    plsey_score = scale_transform(plsey, X, Y)
    spls_score = scale_transform(spls, X, Y)
    assert np.allclose(np.trace(pls_score), np.trace(plsey_score), atol=1e-1)
    assert np.allclose(np.trace(pls_score), np.trace(spls_score), atol=1e-1)


def test_batch_cca():
    epochs = 50
    cca = CCA(latent_dimensions=latent_dims).fit((X, Y))
    ccaey = CCA_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    ccagha = CCA_GHA(
        latent_dimensions=latent_dims,
        epochs=epochs,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    ccasvd = CCA_SVD(
        latent_dimensions=latent_dims,
        epochs=epochs,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    cca_score = cca.score((X, Y))
    ccaey_score = ccaey.score((X, Y))
    ccagh_score = ccagha.score((X, Y))
    ccasvd_score = ccasvd.score((X, Y))
    # check all methods are similar to cca
    assert np.allclose(cca_score.sum(), ccaey_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccagh_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccasvd_score.sum(), atol=1e-1)


def test_stochastic_pls():
    pls = PLS(latent_dimensions=latent_dims).fit((X, Y))
    plsey = PLS_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    spls = PLSStochasticPower(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)

    pls_score = scale_transform(pls, X, Y)
    spls_score = scale_transform(spls, X, Y)
    plsey_score = scale_transform(plsey, X, Y)
    # check all methods are similar to pls
    assert np.allclose(np.trace(pls_score), np.trace(spls_score), atol=1e-1)
    assert np.allclose(np.trace(pls_score), np.trace(plsey_score), atol=1e-1)


def test_stochastic_cca():
    cca = CCA(latent_dimensions=latent_dims).fit((X, Y))
    ccagha = CCA_GHA(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    ccaey = CCA_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    ccasvd = CCA_SVD(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    ).fit((X, Y), **trainer_kwargs)
    cca_score = cca.score((X, Y))
    ccaey_score = ccaey.score((X, Y))
    ccagha_score = ccagha.score((X, Y))
    ccasvd_score = ccasvd.score((X, Y))
    # check all methods are similar to cca
    assert np.allclose(cca_score.sum(), ccaey_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccagha_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccasvd_score.sum(), atol=1e-1)


def test_with_validation():
    cca = CCA(latent_dimensions=latent_dims).fit((X, Y))
    ccaey = CCA_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    ).fit((X, Y), validation_views=(X, Y), **trainer_kwargs)
