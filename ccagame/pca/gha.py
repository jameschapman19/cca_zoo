# Importing necessary libraries

import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from . import _PCA
from .utils import initialize, TV

# Update rule to be used for calculating eigenvectors
from ..utils import data_stream, get_num_batches
import wandb

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
        random_state: int = 0,
        batch_size: int = 128,
        verbose=False,
        wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X):
        U = initialize(
            X, self.n_components, type="random", random_state=self.random_state
        )
        batches = data_stream(X, batch_size=self.batch_size)
        num_batches = get_num_batches(X, batch_size=self.batch_size)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for _ in range(num_batches):
                U = update(U, next(batches), lr=self.lr)
                obj = TV(X, U)
                if self.wandb:
                    wandb.log({"Iteration/Objective": obj})
                else:
                    self.obj.append(obj)
            obj = TV(X, U)
            if self.wandb:
                wandb.log({"Epoch/Objective": obj})
            if self.verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f"epoch {epoch}: {obj}")
        return U


# function form
def calc_gha(
    X, k, lr=1e-1, epochs=100, initialization="uniform", random_state=0, batch_size=None
):
    U = initialize(X, k, type=initialization, random_state=random_state)
    batches = data_stream(X, batch_size=batch_size)
    num_batches = get_num_batches(X, batch_size=batch_size)
    obj = []
    for epoch in range(epochs):
        for _ in range(num_batches):
            U = update(U, next(batches), lr=lr)
        obj.append(TV(X, U))
    return TV(X, U), U, obj
