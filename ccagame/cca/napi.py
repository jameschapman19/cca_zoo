# https://arxiv.org/pdf/1903.08742v1.pdf

# Importing necessary libraries

import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from ccagame.cca.utils import initialize, initialize_gep, gram_schmidt_matrix, TCC
from . import _CCA


# Update rule to be used for calculating eigenvectors
@partial(jit)
def update(A, B, W_, W, lr):
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
            random_state: int = 0,
            verbose=False,
            lr=1,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.epochs = epochs
        self.lr = lr

    def _fit(self, X, Y):
        p = X.shape[1]
        A, B = initialize_gep(X, Y)
        W, V = initialize(
            X, Y, self.n_components, type="random", random_state=self.random_state
        )
        W = jnp.vstack((W, V))
        W_ = jnp.zeros_like(W)
        for epoch in range(self.epochs):
            start_time = time.time()
            W_, W = update(A, B, W_, W, self.lr)
            epoch_time = time.time() - start_time
            if self.verbose:
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f"epoch {epoch}: {TCC(X, Y, W[:p], W[p:])}")
        U = gram_schmidt_matrix(W[:p], B[:p, :p])
        V = gram_schmidt_matrix(W[p:], B[p:, p:])
        return U, V
