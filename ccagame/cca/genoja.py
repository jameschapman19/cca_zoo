"""
Gen-Oja: A Simple and Efficient Algorithm for
Streaming Generalized Eigenvector Computation
https://proceedings.neurips.cc/paper/2018/file/1b318124e37af6d74a03501474f44ea1-Paper.pdf
"""
# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from . import _CCA


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=5)
def update(A, B, W, V, beta, alpha):
    W = W - alpha * (B @ W - A @ V)
    V = V + beta * W
    return W, V / jnp.linalg.norm(V, axis=0)


class Genoja(_CCA):
    def __init__(
        self,
        n_components=4,
        *,
        scale=True,
        copy=True,
        epochs: int = 100,
        random_state: int = None,
        batch_size: int = 128,
        verbose=False,
        beta_0=100,
        alpha=100,
        wandb=False
    ):
        super().__init__(
            n_components,
            scale=scale,
            copy=copy,
            wandb=wandb,
            verbose=verbose,
            random_state=random_state,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta_0 = beta_0
        self.alpha = alpha

    def _fit(self, X, Y, X_val=None, Y_val=None):
        p = X.shape[1]
        A, B = self.initialize_gep(X, Y)
        W, P = self.initialize(
            X, Y, self.n_components, type="random", random_state=self.random_state
        )
        W = jnp.vstack((W, P))
        P = W
        for epoch in range(self.epochs):
            start_time = time.time()
            W, P = update(A, B, W, P, self.beta_0 / (1 + 1e-4 * epoch), self.alpha)
            U = self.gram_schmidt_matrix(W[:p], B[:p, :p])
            V = self.gram_schmidt_matrix(W[p:], B[p:, p:])
            obj_tr = self.TCC(X @ U, Y @ V)
            obj_val = self.TCC(X_val @ W[:p], Y_val @ W[p:])
            self.callback(obj_tr, obj_val, epoch=epoch, start_time=start_time)
        U = self.gram_schmidt_matrix(W[:p], B[:p, :p])
        V = self.gram_schmidt_matrix(W[p:], B[p:, p:])
        return U, V
