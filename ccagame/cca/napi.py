# https://arxiv.org/pdf/1903.08742v1.pdf

# Importing necessary libraries

import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from . import _CCA
from ccagame.solver import agd_solve

# Update rule to be used for calculating eigenvectors
@partial(jit)
def update(A, B, W_, W, lr, solver=agd_solve):
    Z = jnp.linalg.inv(W.T @ B @ W) @ W.T @ A @ W
    W_t1 = solver(A, B, W @ Z)
    W_t1 = W_t1 - lr * W_
    _, R_t1 = jnp.linalg.qr(W_t1)
    R_t1_inv = jnp.linalg.inv(R_t1)
    W_t1 = W_t1 @ R_t1_inv
    W = W @ R_t1_inv
    return W, W_t1


class NAPI(_CCA):
    def __init__(
            self,
            n_components=4,
            *,
            scale=True,
            copy=True,
            epochs: int = 100,
            random_state: int = None,
            verbose=False,
            lr=1,
            wandb=False,
            solver=agd_solve
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.epochs = epochs
        self.lr = lr
        self.solver=solver

    def _fit(self, X, Y, X_val=None, Y_val=None):
        p = X.shape[1]
        A, B = self.initialize_gep(X, Y)
        W, V = self.initialize(
            X, Y, self.n_components, type="random", random_state=self.random_state
        )
        W = jnp.vstack((W, V))
        W_ = jnp.zeros_like(W)
        for epoch in range(self.epochs):
            start_time = time.time()
            W_, W = update(A, B, W_, W, self.lr,self.solver)
            obj_tr = self.TCC(X @ U, Y @ V)
            obj_val = self.TCC(X_val @ U, Y_val @ V)
            self.callback(obj_tr, obj_val, epoch, start_time)
        U = self.gram_schmidt_matrix(W[:p], B[:p, :p])
        V = self.gram_schmidt_matrix(W[p:], B[p:, p:])
        return U, V
