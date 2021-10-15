# TODO
# http://proceedings.mlr.press/v48/aroraa16.pdf

# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
import wandb
from jax import jit
from sklearn.model_selection import train_test_split

from ccagame.utils import data_stream, get_num_batches
from . import _PLS


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(4, 5))
def update(X, Y, U, V, k, lr: float = 0.1):
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
    V :
        all eigenvector estimates for each level
    k :
        number of levels
    lr :
        learning rate
    """
    S = jnp.zeros(min(U.shape[0], V.shape[0]))
    S = S.at[:k].set(1)
    M = U @ jnp.diag(S[:k]) @ V.T + lr * X.T @ Y
    U, _, Vt = jnp.linalg.svd(M)
    return U[:, :k], Vt[:k].T


# Object form
class MSG(_PLS):
    def __init__(
            self,
            n_components=2,
            *,
            scale=True,
            copy=True,
            lr: float = 1,
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

    def _fit(self, X, Y, X_val=None, Y_val=None):
        U, V = self.initialize(X, Y, self.n_components, "random", self.random_state)
        batches = data_stream(X, Y, batch_size=self.batch_size)
        num_batches = get_num_batches(X, Y, batch_size=self.batch_size)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for b in range(num_batches):
                _, (X_i, Y_i) = next(batches)
                U, V = update(X_i, Y_i, U, V, self.n_components, lr=self.lr)
                obj = self.TV(X@U, Y@V)
                if self.wandb:
                    wandb.log({"Iteration/Objective": obj}, step=b)
                else:
                    self.obj.append(obj)
            obj = self.TV(X, Y, U, V)
            if self.wandb:
                wandb.log({"Epoch/Objective": obj}, step=epoch)
            if self.verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f"epoch {epoch} objective: {obj}")
        return U, V
