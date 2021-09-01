# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from ccagame.utils import data_stream, get_num_batches
from . import _PLS
from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(4))
def update(X, Y, U, V, lr: float = 0.1):
    du = X.T @ Y @ V
    uhat = U + lr * du
    return jnp.linalg.qr(uhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_sgd(X, Y, k: int, lr: float = 1, epochs: int = 100,
             random_state: int = 0, batch_size: int = 128):
    """
    Calculate partial least squares weights with SGD method from https://home.ttic.edu/~klivescu/papers/arora_etal_allerton2012.pdf

    Parameters
    ----------
    X :
        First view of data
    Y :
        Second view of data
    k :
        number of latent dimensions
    lr :
        learning rate
    epochs :
        number of epochs
    random_state :
        random seed
    batch_size :
        minibatch size for calculation of stochastic gradients

    Returns
    -------

    """
    U, V = initialize(X, Y, k, 'random', random_state)
    batches = data_stream(X, Y, batch_size=batch_size)
    num_batches = get_num_batches(X, Y, batch_size=batch_size)
    for epoch in range(epochs):
        start_time = time.time()
        for _ in range(num_batches):
            X_i, Y_i = next(batches)
            U = update(X_i, Y_i, U, V, lr=lr)
            V = update(Y_i, X_i, V, U, lr=lr)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} in {epoch_time} sec")
        print(f'epoch {epoch}: {TV(X, Y, U, V)}')
    return TV(X, Y, U, V), U, V


class SGD(_PLS):
    def __init__(self, n_components=2, *, scale=True, copy=True, lr: float = 1, epochs: int = 100,
                 random_state: int = 0, batch_size: int = 128, verbose=False):
        super().__init__(n_components, scale=scale, copy=copy)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, Y):
        U, V = initialize(X, Y, self.n_components, 'random', self.random_state)
        batches = data_stream(X, Y, batch_size=self.batch_size)
        num_batches = get_num_batches(X, Y, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            start_time = time.time()
            for _ in range(num_batches):
                X_i, Y_i = next(batches)
                U = update(X_i, Y_i, U, V, lr=self.lr)
                V = update(Y_i, X_i, V, U, lr=self.lr)
            epoch_time = time.time() - start_time
            if self.verbose:
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f'epoch {epoch}: {TV(X, Y, U, V)}')
        return TV(X, Y, U, V), U, V
