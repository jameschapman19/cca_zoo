# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import grad
from jax import jit

from . import _CCA
from .utils import initialize, TCC
from ..utils import data_stream, get_num_batches


@partial(jit, static_argnums=5)
def model(u, v, X, Y, V, k):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    C_xx = jnp.dot(jnp.transpose(X), X)
    C_yy = jnp.dot(jnp.transpose(Y), Y)
    rewards = (u.T @ C_xy @ v) / (v.T @ C_yy @ v)
    penalties = 0
    for j in range(k):
        penalties = penalties + (u.T @ C_xy @ V[:, j]) ** 2 / (V[:, j].T @ C_yy @ V[:, j])
    return jnp.sum(rewards - penalties) / (u.T @ C_xx @ u)


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=6, static_argnames=('lr', 'riemannian_projection'))
def update(u, v, X, Y, U, V, k, lr: float = 1.0, riemannian_projection=False):
    du = grad(model)(u, v, X, Y, V, k)
    dv = grad(model)(v, u, Y, X, U, k)
    if riemannian_projection:
        dur = du - (u.T @ u) * u
        uhat = u + lr * dur
        dvr = dv - (v.T @ v) * v
        vhat = v + lr * dvr
    else:
        uhat = u + lr * du
        vhat = v + lr * dv
    return uhat / jnp.linalg.norm(uhat), vhat / jnp.linalg.norm(vhat)


# Run the update step iteratively across all eigenvectors
def calc_game(X, Y, k, lr=1, epochs=100, riemannian_projection=False,
              random_state=0, simultaneous=True, batch_size=None):
    U, V = initialize(X, Y, k, 'random', random_state)
    batches = data_stream(X, Y, batch_size=batch_size)
    num_batches = get_num_batches(X, Y, batch_size=batch_size)
    if simultaneous:
        for epoch in range(epochs):
            start_time = time.time()
            for _ in range(num_batches):
                X_i, Y_i = next(batches)
                for k_ in range(k):
                    u, v = update(U[:, k_], V[:, k_], X_i, Y_i, U, V, k_, lr=lr,
                                  riemannian_projection=riemannian_projection)
                    U = U.at[:, k_].set(u)
                    V = V.at[:, k_].set(v)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} in {epoch_time} sec")
            print(f'epoch {epoch}: {TCC(X, Y, U, V)}')
    else:
        for k_ in range(k):
            for epoch in range(epochs):
                start_time = time.time()
                for _ in range(num_batches):
                    X_i, Y_i = next(batches)
                    u, v = update(U[:, k_], V[:, k_], X_i, Y_i, U, V, k_, lr=lr,
                                  riemannian_projection=riemannian_projection)
                    U = U.at[:, k_].set(u)
                    V = V.at[:, k_].set(v)
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f'epoch {epoch}: {TCC(X, Y, U, V)}')
    return TCC(X, Y, U, V), U, V


class Game(_CCA):

    def __init__(self, n_components=4, *, scale=True, copy=True, lr: float = 1.0, epochs: int = 100,
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

    def _fit(self, X, Y):
        U, V = initialize(X, Y, self.n_components, 'random', self.random_state)
        batches = data_stream(X, Y, batch_size=self.batch_size)
        num_batches = get_num_batches(X, Y, batch_size=self.batch_size)
        if self.simultaneous:
            for epoch in range(self.epochs):
                start_time = time.time()
                for _ in range(num_batches):
                    X_i, Y_i = next(batches)
                    for k_ in range(self.n_components):
                        u, v = update(U[:, k_], V[:, k_], X_i, Y_i, U, V, k_, lr=self.lr,
                                      riemannian_projection=self.riemannian_projection)
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                epoch_time = time.time() - start_time
                if self.verbose:
                    print(f"Epoch {epoch} in {epoch_time} sec")
                    print(f'epoch {epoch}: {TCC(X, Y, U, V)}')
        else:
            for k_ in range(self.n_components):
                for epoch in range(self.epochs):
                    start_time = time.time()
                    for _ in range(num_batches):
                        X_i, Y_i = next(batches)
                        u, v = update(U[:, k_], V[:, k_], X_i, Y_i, U, V, k_, lr=self.lr,
                                      riemannian_projection=self.riemannian_projection)
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                    epoch_time = time.time() - start_time
                    if self.verbose:
                        print(f"Epoch {epoch} in {epoch_time} sec")
                        print(f'epoch {epoch}: {TCC(X, Y, U, V)}')
        return U, V
