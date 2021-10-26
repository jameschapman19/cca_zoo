# Importing necessary libraries

import time
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from ccagame.utils import data_stream, get_num_batches
from . import _PLS


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(5))
def update(X, Y, U, S, V, k):
    """
    Update the left and right singular vector estimates

    Parameters
    ----------
    X :
        batch of data for view X
    Y :
        batch of data for view Y
    U :
        all eigenvector estimates for each level
    S :
        all singular value estimates
    V :
        all eigenvector estimates for each level
    k :
        number of levels
    """
    uhat = X @ U
    u_orth = X - X @ U @ U.T
    vhat = Y @ V
    v_orth = Y - Y @ V @ V.T
    Q = jnp.vstack(
        (
            jnp.hstack((jnp.diag(S) + uhat.T @ vhat, jnp.linalg.norm(v_orth) * uhat.T)),
            jnp.hstack(
                (
                    jnp.linalg.norm(u_orth) * vhat,
                    jnp.atleast_2d(jnp.linalg.norm(u_orth) * jnp.linalg.norm(v_orth)),
                )
            ),
        )
    )
    U_, S, V_ = jnp.linalg.svd(Q)
    U = jnp.hstack((U, u_orth.T / jnp.linalg.norm(u_orth))) @ U_[:, :k]
    V = jnp.hstack((V, v_orth.T / jnp.linalg.norm(v_orth))) @ V_.T[:, :k]
    return U, S[:k], V


# Object form
class Incremental(_PLS):
    def __init__(
            self,
            n_components=2,
            *,
            scale=True,
            copy=True,
            lr: float = 1,
            epochs: int = 100,
            random_state: int = None,
            verbose=False,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.lr = lr
        self.epochs = epochs

    def _fit(self, X, Y, X_val=None, Y_val=None):
        U, V = self.initialize(X, Y, self.n_components, "random", self.random_state)
        batches = data_stream(X, Y, batch_size=1)
        num_batches = get_num_batches(X, Y, batch_size=1)
        S = np.zeros(self.n_components)
        self.obj = []
        for epoch in range(self.epochs):
            obj_tr = 0
            obj_val = 0
            start_time = time.time()
            for b in range(num_batches):
                _, (X_i, Y_i) = next(batches)
                U, S, V = update(X_i, Y_i, U, S, V, self.n_components)
                obj_tr += self.TV(X @ U, Y @ V)
                obj_val += self.TV(X_val @ U, Y_val @ V)
            self.callback(obj_tr/num_batches, obj_val/num_batches, epoch=epoch, start_time=start_time)
        return U, V
