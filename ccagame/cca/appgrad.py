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

from . import _CCA


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
        random_state: int = None,
        verbose=False,
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

    def _fit(self, X, Y, X_val=None, Y_val=None):
        U, V = self.initialize(X, Y, self.n_components, "random", self.random_state)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            U, V, phi_, psi_ = update(U, V, phi_, psi_)
            obj_tr = self.TCC(X @ U, Y @ V)
            obj_val = self.TCC(X_val @ U, Y_val @ V)
            self.callback(obj_tr, obj_val, epoch=epoch, start_time=start_time)
        return U, V
