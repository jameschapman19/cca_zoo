"""
Efficient Algorithms for Large-scale Generalized Eigenvector
Computation and Canonical Correlation Analysis
https://export.arxiv.org/pdf/1604.03930
"""
# Importing necessary libraries
import time

import jax.numpy as jnp
from jax import random, jit

from ccagame.solver import agd_solve
from . import _CCA
from .utils import gram_schmidt_matrix, initialize_gep, initialize, TCC
from functools import partial


# Update rule to be used for calculating eigenvectors
@partial(jit)
def update(X, Y, phi, psi, phi_, psi_, lr):
    n = X.shape[0]
    phi_t = phi_ - lr * X.T @ (X @ phi_ - Y @ psi) / n
    U, S, Vt = jnp.linalg.svd(phi_.T @ X.T @ X @ phi_)
    phit = phi_ @ U @ jnp.diag(S ** -0.5) @ phi_.T
    psi_t = psi_ - lr * Y.T @ (Y @ psi_ - X @ phi) / n
    U, S, Vt = jnp.linalg.svd(psi_.T @ Y.T @ Y @ psi_)
    psit = psi_ @ U @ jnp.diag(S ** -0.5) @ psi_.T
    return phit, psit, phi_t, psi_t


class AppGrad(_CCA):
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
        U, V = initialize(X, Y, self.n_components, "random", self.random_state)
        for epoch in range(self.epochs):
            start_time = time.time()
            U, V, phi_, psi_ = update(U, V, phi_, psi_)
            epoch_time = time.time() - start_time
            if self.verbose:
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f"epoch {epoch}: {TCC(X, Y, W[:p], W[p:])}")
        U = gram_schmidt_matrix(W[:p], B[:p, :p])
        V = gram_schmidt_matrix(W[p:], B[p:, p:])
        return U, V
