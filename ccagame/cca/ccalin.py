"""
Efficient Algorithms for Large-scale Generalized Eigenvector
Computation and Canonical Correlation Analysis
https://export.arxiv.org/pdf/1604.03930
"""
# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from ccagame.solver import agd_solve
from . import _CCA
from .utils import gram_schmidt_matrix, initialize_gep, initialize, TCC


def obj(W, A, B, Wt):
    return 0.5 * jnp.sum(jnp.diag(W.T @ B @ W - W.T @ A @ Wt))

def update(A, B, W, solver=agd_solve, **kwargs):
    gamma = jnp.linalg.inv(W.T @ B @ W) @ W.T @ A @ W
    W = solver(obj, A, B, W, x=W @ gamma, **kwargs)
    W = gram_schmidt_matrix(W, B)
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
            wandb=False,
            solver=agd_solve
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.epochs = epochs
        self.solver = solver

    def _fit(self, X, Y):
        p = X.shape[1]
        A, B = initialize_gep(X, Y)
        W, V = initialize(
            X, Y, self.n_components, type="random", random_state=self.random_state
        )
        W = jnp.vstack((W, V))
        for epoch in range(self.epochs):
            start_time = time.time()
            W = update(A, B, W)
            epoch_time = time.time() - start_time
            if self.verbose:
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f"epoch {epoch}: {TCC(X, Y, W[:p], W[p:])}")
        U = gram_schmidt_matrix(W[:p], B[:p, :p])
        V = gram_schmidt_matrix(W[p:], B[p:, p:])
        return U, V
