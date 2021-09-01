# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from ccagame.utils import data_stream, get_num_batches
from . import _PLS
from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
@partial(jit)
def update(X, Y, V):
    U = X.T @ Y @ V
    V = Y.T @ X @ U
    return jnp.linalg.qr(U)[0], jnp.linalg.qr(V)[0]


# Run the update step iteratively across all eigenvectors
def calc_batch(X, Y, k: int, epochs: int = 100,
               random_state: int = 0):
    """
    Calculate partial least squares weights with batch power method

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
    batches = data_stream(X, Y, batch_size=None)
    num_batches = get_num_batches(X, Y, batch_size=None)
    for epoch in range(epochs):
        start_time = time.time()
        for _ in range(num_batches):
            U, V = update(*next(batches), V)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} in {epoch_time} sec")
        print(f'epoch {epoch}: {TV(X, Y, U, V)}')
    return TV(X, Y, U, V), U, V


class Batch(_PLS):
    def __init__(self, n_components=2, *, scale=True, copy=True, lr: float = 1, epochs: int = 100,
                 random_state: int = 0, verbose=False):
        super().__init__(n_components, scale=scale, copy=copy)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, X, Y):
        U, V = initialize(X, Y, self.n_components, 'random', self.random_state)
        batches = data_stream(X, Y, batch_size=None)
        num_batches = get_num_batches(X, Y, batch_size=None)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for _ in range(num_batches):
                U, V = update(*next(batches), V)
                self.obj.append(TV(X, Y, U, V))
            epoch_time = time.time() - start_time
            if self.verbose:
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f'epoch {epoch}: {TV(X, Y, U, V)}')
        return U, V
