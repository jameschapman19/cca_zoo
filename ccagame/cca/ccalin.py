"""
Efficient Algorithms for Large-scale Generalized Eigenvector
Computation and Canonical Correlation Analysis
https://export.arxiv.org/pdf/1604.03930
"""
# Importing necessary libraries
import jax.numpy as jnp
from jax import random, jit

from ccagame.solver import agd_solve
from . import _CCA
from .utils import gram_schmidt_matrix, initialize_gep


@jit
def obj(W, A, B, Wt):
    return jnp.trace(0.5 * W.T @ B @ W - W.T @ A @ Wt)


@jit
def gamma(W, A):
    return W.T @ A @ W


def GenELinK_update(W, A, B, lr, mu, iterations):
    W = jnp.squeeze(
        agd_solve(
            obj,
            A,
            B,
            W,
            x=jnp.expand_dims(jnp.dot(W, gamma(W, A)), 0),
            lr=lr,
            mu=mu,
            iterations=iterations,
            in_axes=(0, None, None, None),
        ),
        0,
    )
    return gram_schmidt_matrix(W, B)


def GenELinK(A, B, X, Y, k, epochs=1000, random_state=0):
    p = X.shape[1]
    d = A.shape[1]
    key = random.PRNGKey(random_state)
    beta = jnp.linalg.norm(B)
    alpha = jnp.min(jnp.abs(jnp.linalg.eig(B)[0]))
    Q = beta / alpha
    mu = (jnp.sqrt(Q) - 1) / (jnp.sqrt(Q) + 1)
    lr = 1 / beta
    W = random.normal(key, (d, k))
    W = gram_schmidt_matrix(W, B)
    for i in range(epochs):
        W = GenELinK_update(W, A, B, lr, mu, epochs)
    return W


class CCALin(_CCA):
    def __init__(
            self,
            n_components=2,
            *,
            scale=True,
            copy=True,
            epochs: int = 100,
            random_state: int = 0,
            verbose=False,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb)
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, X, Y):
        p = X.shape[1]
        A, B = initialize_gep(X, Y)
        W = GenELinK(
            A,
            B,
            X,
            Y,
            2 * self.n_components,
            epochs=self.epochs,
            random_state=self.random_state,
        )
        key = random.PRNGKey(self.random_state)
        M = random.normal(key, (2 * self.n_components, self.n_components))
        U = gram_schmidt_matrix(jnp.dot(W[:p], M), B[:p, :p])
        V = gram_schmidt_matrix(jnp.dot(W[p:], M), B[p:, p:])
        return U, V
