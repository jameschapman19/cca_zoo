# Importing necessary libraries

import time
from functools import partial

import jax.numpy as jnp
import numpy as np
import wandb
from jax import jit
from sklearn.model_selection import train_test_split

from ccagame.utils import data_stream, get_num_batches
from . import _PLS
from .utils import TV, initialize


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
            random_state: int = 0,
            verbose=False,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, X, Y):
        X, X_val, Y, Y_val = train_test_split(
            X, Y, random_state=self.random_state, train_size=0.9
        )
        U, V = initialize(X, Y, self.n_components, "random", self.random_state)
        batches = data_stream(X, Y, batch_size=1)
        num_batches = get_num_batches(X, Y, batch_size=1)
        S = np.zeros(self.n_components)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for b in range(num_batches):
                U, S, V = update(*next(batches), U, S, V, self.n_components)
                obj = TV(X, Y, U, V)
                if self.wandb:
                    wandb.log({"Iteration/Objective": obj}, step=b)
                else:
                    self.obj.append(obj)
            obj = TV(X, Y, U, V)
            if self.wandb:
                wandb.log({"Epoch/Objective": obj}, step=epoch)
            if self.verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f"epoch {epoch} objective: {obj}")
        return U, V
