# Importing necessary libraries

import time
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from ccagame.utils import data_stream, get_num_batches
from . import _PLS
from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(5))
def update(X, Y, U, S, V, k):
    uhat = X @ U
    u_orth = X - X @ U @ U.T
    vhat = Y @ V
    v_orth = Y - Y @ V @ V.T
    Q = jnp.vstack((jnp.hstack((jnp.diag(S) + uhat.T @ vhat, jnp.linalg.norm(v_orth) * uhat.T)),
                    jnp.hstack((jnp.linalg.norm(u_orth) * vhat,
                                jnp.atleast_2d(jnp.linalg.norm(u_orth) * jnp.linalg.norm(v_orth))))))
    U_, S, V_ = jnp.linalg.svd(Q)
    U = jnp.hstack((U, u_orth.T / jnp.linalg.norm(u_orth))) @ U_[:, :k]
    V = jnp.hstack((V, v_orth.T / jnp.linalg.norm(v_orth))) @ V_.T[:, :k]
    return U, S[:k], V


# Run the update step iteratively across all eigenvectors
def calc_incremental(X, Y, k: int, epochs: int = 100,
                     random_state: int = 0):
    """
    Calculate partial least squares weights with incremental method from https://home.ttic.edu/~klivescu/papers/arora_etal_allerton2012.pdf

    Parameters
    ----------
    X :
        First view of data
    Y :
        Second view of data
    k :
        number of latent dimensions
    epochs :
        number of epochs
    random_state :
        random seed

    Returns
    -------

    """
    U, V = initialize(X, Y, k, 'random', random_state)
    batches = data_stream(X, Y, batch_size=1)
    num_batches = get_num_batches(X, Y, batch_size=1)
    S = np.zeros(k)
    for epoch in range(epochs):
        start_time = time.time()
        for _ in range(num_batches):
            U, S, V = update(*next(batches), U, S, V, k)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} in {epoch_time} sec")
        print(f'epoch {epoch}: {TV(X, Y, U, V)}')
    return TV(X, Y, U, V), U, V


class Incremental(_PLS):
    def __init__(self, n_components=2, *, scale=True, copy=True, lr: float = 1, epochs: int = 100,
                 random_state: int = 0, verbose=False):
        super().__init__(n_components, scale=scale, copy=copy)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, X, Y):
        U, V = initialize(X, Y, self.n_components, 'random', self.random_state)
        batches = data_stream(X, Y, batch_size=1)
        num_batches = get_num_batches(X, Y, batch_size=1)
        S = np.zeros(self.n_components)
        for epoch in range(self.epochs):
            start_time = time.time()
            for _ in range(num_batches):
                U, S, V = update(*next(batches), U, S, V, self.n_components)
            epoch_time = time.time() - start_time
            if self.verbose:
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f'epoch {epoch}: {TV(X, Y, U, V)}')
        return U, V
