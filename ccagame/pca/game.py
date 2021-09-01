"""
EigenGame: PCA as a Nash Equilibrium
https://arxiv.org/pdf/2010.00554.pdf
"""
import time
# Importing necessary libraries
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from . import _PCA
from .utils import TV, initialize
# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V holds the previously computed eigenvectors
from ..utils import data_stream, get_num_batches


@partial(jit, static_argnums=(3))
def alpha_model(u, X, U, k):
    n = X.shape[0]
    M = X.T @ X / n
    rewards = u.T @ M @ u
    penalties = 0
    for j in range(k):
        penalties = penalties + (u.T @ M @ U[:, j]) ** 2 / (
                U[:, j].T @ M @ U[:, j])
    return jnp.sum(rewards - penalties)


@partial(jit, static_argnums=(3))
def mu_model(u, X, U, k):
    M = X.T @ X
    rewards = u.T @ M @ u
    penalties = 0
    for j in range(k):
        penalties = penalties + (u.T @ M @ U[:, j]) ** 2 / (
                U[:, j].T @ M @ U[:, j])
    return jnp.sum(rewards - penalties)


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=3, static_argnames=('lr', 'riemannian_projection', 'mu'))
def update(u, X, U, k, lr:float=1.0, riemannian_projection=False, mu=False):
    if mu:
        du = mu_model(u, X, U, k)
    else:
        du = jax.grad(alpha_model)(u, X, U, k)
    if riemannian_projection:
        dur = du - (u.T @ u) * u
        uhat = u + lr * dur
    else:
        uhat = u + lr * du
    return uhat / jnp.linalg.norm(uhat)


def calc_game(X, k, lr: float = 1.0, epochs=100, riemannian_projection=False, initialization='random',
              random_state=0, simultaneous=False, batch_size=None):
    U = initialize(X, k, type=initialization, random_state=random_state)
    batches = data_stream(X, batch_size=batch_size)
    num_batches = get_num_batches(X, batch_size=batch_size)
    obj = []
    if simultaneous:
        for epoch in range(epochs):
            start_time = time.time()
            for _ in range(num_batches):
                X_i = next(batches)
                for k_ in range(k):
                    u = update(U[:, k_], X_i, U, k_, lr=lr, riemannian_projection=riemannian_projection)
                    U = U.at[:, k_].set(u)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} in {epoch_time} sec")
            obj.append(TV(X, U))
            print(f'epoch {epoch}: {obj[-1]}')
    else:
        for k_ in range(k):
            for epoch in range(epochs):
                start_time = time.time()
                for _ in range(num_batches):
                    X_i = next(batches)
                    u = update(U[:, k_], X_i, U, k_, lr=lr, riemannian_projection=riemannian_projection)
                    U = U.at[:, k_].set(u)
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch} in {epoch_time} sec")
                obj.append(TV(X, U))
                print(f'epoch {epoch}: {obj[-1]}')
    return TV(X, U), U, obj


class Game(_PCA):

    def __init__(self, n_components=4, *, scale=True, copy=True, lr: float = 1, epochs: int = 100,
                 riemannian_projection: bool = False,
                 random_state: int = 0, simultaneous: bool = True, batch_size: int = 128, mu=True, verbose=False):
        super().__init__(n_components, scale=scale, copy=copy)
        self.lr = lr
        self.epochs = epochs
        self.riemannian_projection = riemannian_projection
        self.random_state = random_state
        self.simultaneous = simultaneous
        self.batch_size = batch_size
        self.mu = mu
        self.verbose = verbose

    def _fit(self, X):
        U = initialize(X, self.n_components, type='random', random_state=self.random_state)
        batches = data_stream(X, batch_size=self.batch_size)
        num_batches = get_num_batches(X, batch_size=self.batch_size)
        obj = []
        if self.simultaneous:
            for epoch in range(self.epochs):
                start_time = time.time()
                for _ in range(num_batches):
                    X_i = next(batches)
                    for k_ in range(self.n_components):
                        u = update(U[:, k_], X_i, U, k_, lr=self.lr, riemannian_projection=self.riemannian_projection)
                        U = U.at[:, k_].set(u)
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch} in {epoch_time} sec")
                obj.append(TV(X, U))
                print(f'epoch {epoch}: {obj[-1]}')
        else:
            for k_ in range(self.n_components):
                for epoch in range(self.epochs):
                    start_time = time.time()
                    for _ in range(num_batches):
                        X_i = next(batches)
                        u = update(U[:, k_], X_i, U, k_, lr=self.lr, riemannian_projection=self.riemannian_projection)
                        U = U.at[:, k_].set(u)
                    epoch_time = time.time() - start_time
                    print(f"Epoch {epoch} in {epoch_time} sec")
                    obj.append(TV(X, U))
                    print(f'epoch {epoch}: {obj[-1]}')
        return U
