"""
On Constrained Nonconvex Stochastic Optimization: A Case
Study for Generalized Eigenvalue Decomposition
https://proceedings.mlr.press/v89/chen19a/chen19a.pdf
"""
# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from . import _CCA
import wandb

# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=5)
def update(A, B, W, lr):
    Y = W.T @ A @ W
    W = W - lr * (B @ W @ Y - A @ W)
    return W


class Lagrange(_CCA):
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
            lr=1,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def _fit(self, X, Y, X_val=None, Y_val=None):
        p = X.shape[1]
        A, B = self.initialize_gep(X, Y)
        W, V = self.initialize(
            X, Y, self.n_components, type="random", random_state=self.random_state
        )
        W = jnp.vstack((W, V))
        for epoch in range(self.epochs):
            start_time = time.time()
            W = update(A, B, W, self.lr)
            obj_tr = self.TV(X @ W[:p], Y @ W[p:])
            obj_val = self.TV(X_val @ W[:p], Y_val @ W[p:])
            if self.wandb:
                wandb.log({"Iteration/Objective (Train)": obj_tr,
                           "Iteration/Objective (Val)": obj_val}, step=epoch)
            else:
                self.obj.append([obj_tr, obj_val])
            if self.verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f"Epoch {epoch} objective (Train): {obj_tr}")
                print(f"Epoch {epoch} objective (Train): {obj_val}")
        U = self.gram_schmidt_matrix(W[:p], B[:p, :p])
        V = self.gram_schmidt_matrix(W[p:], B[p:, p:])
        return U, V
