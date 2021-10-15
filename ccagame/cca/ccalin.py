"""
Efficient Algorithms for Large-scale Generalized Eigenvector
Computation and Canonical Correlation Analysis
https://export.arxiv.org/pdf/1604.03930
"""
# Importing necessary libraries
import time

import jax.numpy as jnp

from ccagame.solver import agd_solve
from . import _CCA
import wandb

def obj(W, A, B, Wt):
    return 0.5 * jnp.sum(jnp.diag(W.T @ B @ W - W.T @ A @ Wt))


def update(A, B, W, solver=agd_solve, **kwargs):
    gamma = jnp.linalg.inv(W.T @ B @ W) @ W.T @ A @ W
    W = solver(obj, A, B, W, x=W @ gamma, **kwargs)
    return W


class CCALin(_CCA):
    def __init__(
            self,
            n_components=2,
            *,
            scale=True,
            copy=True,
            epochs: int = 100,
            random_state: int = None,
            verbose=False,
            wandb=False,
            solver=agd_solve
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.epochs = epochs
        self.solver = solver

    def _fit(self, X, Y, X_val=None, Y_val=None):
        p = X.shape[1]
        A, B = self.initialize_gep(X, Y)
        W, V = self.initialize(
            X, Y, self.n_components, type="random", random_state=self.random_state
        )
        W = jnp.vstack((W, V))
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            W = update(A, B, W)
            W = self.gram_schmidt_matrix(W, B)
            obj_tr = self.TCC(X @ W[:p], Y @ W[p:])
            obj_val = self.TCC(X_val @ W[:p], Y_val @ W[p:])
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
