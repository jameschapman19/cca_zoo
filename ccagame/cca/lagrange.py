"""
On Constrained Nonconvex Stochastic Optimization: A Case
Study for Generalized Eigenvalue Decomposition
https://proceedings.mlr.press/v89/chen19a/chen19a.pdf
"""
# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit

from ccagame.cca.utils import initialize, initialize_gep, gram_schmidt_matrix, TCC
from . import _CCA


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=5)
def update(A, B, W, lr):
    Y = jnp.dot(jnp.transpose(W), jnp.dot(A, W))
    W = W - lr * (B @ W @ Y - A @ W)
    return W


# Run the update step iteratively across all eigenvectors
def calc_lagrangeminmax(X, Y, k, iterations=100, lr=100, random_state=0):
    p = X.shape[1]
    A, B = initialize_gep(X, Y)
    W, V = initialize(X, Y, k, type="random", random_state=random_state)
    W = jnp.vstack((W, V))
    for i in range(iterations):
        W = update(A, B, W, lr)
    Wx = gram_schmidt_matrix(W[:p], B[:p, :p])
    Wy = gram_schmidt_matrix(W[p:], B[p:, p:])
    return TCC(X, Y, W[:p], W[p:]), Wx, Wy


class Lagrange(_CCA):
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
        lr=1,
    ):
        super().__init__(n_components, scale=scale, copy=copy)
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.verbose = verbose
        self.lr = lr

    def _fit(self, X, Y):
        p = X.shape[1]
        A, B = initialize_gep(X, Y)
        W, V = initialize(
            X, Y, self.n_components, type="random", random_state=self.random_state
        )
        W = jnp.vstack((W, V))
        for i in range(self.epochs):
            W = update(A, B, W, self.lr)
            print(f"iteration {i}: {TCC(X, Y, W[:p], W[p:])}")
        U = gram_schmidt_matrix(W[:p], B[:p, :p])
        V = gram_schmidt_matrix(W[p:], B[p:, p:])
        return U, V
