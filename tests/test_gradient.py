import numpy as np
import pytest
from sklearn.utils import check_random_state

from cca_zoo.linear import CCA, PLS
from cca_zoo.linear._gradient._svd import CCA_SVD

n = 50
rng = check_random_state(0)
X = rng.rand(n, 4)
Y = rng.rand(n, 6)
Z = rng.rand(n, 8)
# centre the data
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)
Z -= Z.mean(axis=0)

latent_dims = 3
epochs = 200
batch_size = 5
random_state = 1
trainer_kwargs = dict(
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)


def scale_transform(model, X, Y):
    Zx, Zy = model.transform((X, Y))
    Zx /= np.linalg.norm(model.weights_[0], axis=0, keepdims=True)
    Zy /= np.linalg.norm(model.weights_[1], axis=0, keepdims=True)
    return np.abs(np.cov(Zx, Zy, rowvar=False)[:latent_dims, latent_dims:])


def test_batch_pls():
    pytest.importorskip("torch")
    from torch import manual_seed

    from cca_zoo.linear import PLS_EY  # , PLSStochasticPower

    pls = PLS(latent_dimensions=3).fit((X, Y))
    manual_seed(42)
    plsey = PLS_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y))
    pls_score = scale_transform(pls, X, Y)
    plsey_score = scale_transform(plsey, X, Y)
    assert np.allclose(np.trace(pls_score), np.trace(plsey_score), atol=1e-2)


def test_batch_cca():
    pytest.importorskip("torch")
    from cca_zoo.linear import CCA_EY, CCA_GHA

    cca = CCA(latent_dimensions=3).fit((X, Y))
    ccaey = CCA_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y))
    ccagha = CCA_GHA(
        latent_dimensions=latent_dims,
        epochs=epochs,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y))
    # ccasvd = CCA_SVD(
    #     latent_dimensions=latent_dims,
    #     epochs=epochs,
    #     random_state=random_state,
    #     trainer_kwargs=trainer_kwargs,
    # ).fit((X, Y))
    cca_score = cca.score((X, Y))
    ccaey_score = ccaey.score((X, Y))
    ccagh_score = ccagha.score((X, Y))
    # ccasvd_score = ccasvd.score((X, Y))
    # check all methods are similar to cca
    assert np.allclose(cca_score.sum(), ccaey_score.sum(), atol=1e-1)
    assert np.allclose(cca_score.sum(), ccagh_score.sum(), atol=1e-1)
    # assert np.allclose(cca_score.sum(), ccasvd_score.sum(), atol=1e-1)


def test_stochastic_pls():
    pytest.importorskip("torch")
    from torch import manual_seed

    from cca_zoo.linear import PLS_EY, PLSStochasticPower

    pls = PLS(latent_dimensions=3).fit((X, Y))
    manual_seed(42)
    plsey = PLS_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y))
    spls = PLSStochasticPower(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y))

    pls_score = scale_transform(pls, X, Y)
    spls_score = scale_transform(spls, X, Y)
    plsey_score = scale_transform(plsey, X, Y)
    # check all methods are similar to pls
    assert np.allclose(np.trace(pls_score), np.trace(spls_score), atol=1e-1)
    assert np.allclose(np.trace(pls_score), np.trace(plsey_score), atol=1e-1)


def test_stochastic_cca():
    pytest.importorskip("torch")
    from cca_zoo.linear import CCA_EY, CCA_GHA

    cca = CCA(latent_dimensions=3).fit((X, Y))
    ccagha = CCA_GHA(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y))
    ccaey = CCA_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y))
    ccasvd = CCA_SVD(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y))
    cca_score = cca.score((X, Y))
    ccaey_score = ccaey.score((X, Y))
    ccagha_score = ccagha.score((X, Y))
    ccasvd_score = ccasvd.score((X, Y))
    # check all methods are similar to cca
    assert np.allclose(cca_score.sum(), ccaey_score.sum(), atol=2e-1)
    assert np.allclose(cca_score.sum(), ccagha_score.sum(), atol=2e-1)
    assert np.allclose(cca_score.sum(), ccasvd_score.sum(), atol=2e-1)


def test_with_validation():
    pytest.importorskip("torch")
    from cca_zoo.linear import CCA_EY

    cca = CCA(latent_dimensions=3).fit((X, Y))
    ccaey = CCA_EY(
        latent_dimensions=latent_dims,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        trainer_kwargs=trainer_kwargs,
    ).fit((X, Y), validation_views=(X, Y))
