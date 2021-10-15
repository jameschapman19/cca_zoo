"""
Exponentially convergent stochastic k-PCA without variance reduction
https://arxiv.org/pdf/1904.01750.pdf
"""
import time
# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
import wandb
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
    du = (X - X @ U @ U.T).T @ X @ U
    vhat = U + lr * du
    return jnp.linalg.qr(vhat)[0]


# object form
class Krasulina(_PCA):
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
                _, X_i = next(batches)
                U = update(U, X_i, lr=self.lr)
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
