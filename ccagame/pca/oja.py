# Importing necessary libraries

import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from . import _PCA
from .utils import TV, initialize
# Update rule to be used for calculating eigenvectors
from ..utils import data_stream, get_num_batches


@partial(jit, static_argnums=(2))
def update(u, X, lr=0.1):
    dv = X.T @ X @ u
    vhat = u + lr * dv
    return jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_oja(X, k, lr=1e-1, epochs=100,
             random_state=0, batch_size=None):
    U = initialize(X, k, type='random', random_state=random_state)
    batches = data_stream(X, batch_size=batch_size)
    num_batches = get_num_batches(X, batch_size=batch_size)
    obj = []
    for epoch in range(epochs):
        start_time = time.time()
        for _ in range(num_batches):
            U = update(U, next(batches), lr=lr)
        epoch_time = time.time() - start_time
        obj.append(TV(X, U))
        print(f"Epoch {epoch} in {epoch_time} sec")
        print(f'epoch {epoch}: {obj[-1]}')
    return TV(X, U), U, obj


class Oja(_PCA):
    def __init__(self, n_components=2, *, scale=True, copy=True, lr: float = 1e-2, epochs: int = 100,
                 random_state: int = 0, batch_size: int = 128, verbose=False):
        super().__init__(n_components, scale=scale, copy=copy)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X):
        U = initialize(X, self.n_components, type='random', random_state=self.random_state)
        batches = data_stream(X, batch_size=self.batch_size)
        num_batches = get_num_batches(X, batch_size=self.batch_size)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for _ in range(num_batches):
                U = update(U, next(batches), lr=self.lr)
                self.obj.append(TV(X, U))
            epoch_time = time.time() - start_time
            if self.verbose:
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f'epoch {epoch}: {TV(X, U)}')
        return U
