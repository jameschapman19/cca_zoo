# Importing necessary libraries

import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from . import _PCA
# Update rule to be used for calculating eigenvectors
from ..utils import data_stream, get_num_batches


@partial(jit, static_argnums=(2))
def update(U, X, lr=0.1):
    """
    Update all of the singular vector estimates

    Parameters
    ----------
    X :
        batch of data for view X
    U :
        all eigenvector estimates for each level
    lr :
        learning rate
    """
    dv = X.T @ X @ U - U @ jnp.triu(U.T @ X.T @ X @ U)
    vhat = U + lr * dv
    return vhat / jnp.linalg.norm(vhat, axis=0)


# Object form
class GHA(_PCA):
    def __init__(
            self,
            n_components=2,
            *,
            scale=True,
            copy=True,
            lr: float = 1e-2,
            epochs: int = 100,
            random_state: int = None,
            batch_size: int = 128,
            verbose=False,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def _fit(self, X, X_val):
        U = self.initialize(
            X, self.n_components, type="random", random_state=self.random_state
        )
        batches = data_stream(X, batch_size=self.batch_size)
        num_batches = get_num_batches(X, batch_size=self.batch_size)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for b in range(num_batches):
                _, X = next(batches)
                U = update(U, X, lr=self.lr)
                obj_tr = self.TV(X @ U)
                obj_val = self.TV(X_val @ U)
                self.callback(obj_tr, obj_val)
            self.callback(obj_tr, obj_val, epoch=epoch, start_time=start_time)
        return U
