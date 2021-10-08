# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
import wandb
from jax import jit

from ccagame.utils import data_stream, get_num_batches
from . import _PLS
from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
@partial(jit)
def update(X, Y, V):
    """
    Update the left and right singular vector estimates by one iteration of the power method. Could be stochastic if X and Y are batches rather than full data

    Parameters
    ----------
    X :
        batch of data for view X
    Y :
        batch of data for view Y
    V :
        all eigenvector estimates for each level
    """
    U = X.T @ Y @ V
    V = Y.T @ X @ U
    return jnp.linalg.qr(U)[0], jnp.linalg.qr(V)[0]


# Object form
class Batch(_PLS):
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
        U, V = initialize(X, Y, self.n_components, "random", self.random_state)
        batches = data_stream(X, Y, batch_size=None)
        num_batches = get_num_batches(X, Y, batch_size=None)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for b in range(num_batches):
                _, (X_i, Y_i) = next(batches)
                U, V = update(X_i,Y_i, V)
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
