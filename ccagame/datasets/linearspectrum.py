import jax.numpy as jnp
from ccagame.utils import data_stream
import numpy as np
from cca_zoo.models import rCCA


def linear_spectrum():
    pass


def linear_spectrum_iterator(batch_size, n_components, pca=False, cca=False):
    X, Y, X_te, Y_te = linear_spectrum()
    if cca:
        cca = rCCA(latent_dims=n_components, scale=False, centre=False, c=0.01).fit(
            (X, Y)
        )
        correct_U, correct_V = cca.weights
        correct_U /= jnp.linalg.norm(correct_U, axis=0)
        correct_V /= jnp.linalg.norm(correct_V, axis=0)
    else:
        correct_U, _, correct_V = jnp.linalg.svd(X.T @ Y)
        correct_U = correct_U[:, :n_components]
        correct_V = correct_V[:n_components, :].T  # cca.score((X,Y))
    return (
        data_stream(X, Y=Y, batch_size=batch_size),
        (X_te, Y_te),
        (correct_U, correct_V),
        (X.shape[1], Y.shape[1]),
    )
