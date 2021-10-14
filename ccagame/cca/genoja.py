"""
Gen-Oja: A Simple and Efficient Algorithm for
Streaming Generalized Eigenvector Computation
https://proceedings.neurips.cc/paper/2018/file/1b318124e37af6d74a03501474f44ea1-Paper.pdf
"""
# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit

from ccagame.cca.utils import initialize, initialize_gep, gram_schmidt_matrix, TCC
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
            random_state: int = 0,
            batch_size: int = 128,
            verbose=False,
            beta_0,
            alpha,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta_0 = beta_0
        self.alpha = alpha

    def _fit(self, X, Y):
        p = X.shape[1]
        A, B = initialize_gep(X, Y)
        W, P = initialize(
            X, Y, self.n_components, type="random", random_state=self.random_state
        )
        W = jnp.vstack((W, P))
        P = W
        for i in range(self.epochs):
            W, P = update(A, B, W, P, self.beta_0 / (1 + 1e-4 * i), self.alpha)
            U = gram_schmidt_matrix(W[:p], B[:p, :p])
            V = gram_schmidt_matrix(W[p:], B[p:, p:])
            if self.verbose:
                print(f"iteration {i}: {TCC(X, Y, U, V)}")
        return U, V
