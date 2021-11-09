# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import jit

from ccagame.utils import data_stream, get_num_batches
from . import _PLS


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(4))
def update(X, Y, U, V, lr: float = 0.1):
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
    lr :
        learning rate
    """
    du = X.T @ Y @ V
    uhat = U + lr * du
    return jnp.linalg.qr(uhat)[0]


# Object form
class SGD(_PLS):
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
        super().__init__(
            n_components,
            scale=scale,
            copy=copy,
            wandb=wandb,
            verbose=verbose,
            random_state=random_state,
        )
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
                U = update(X_i, Y_i, U, V, lr=self.lr)
                V = update(Y_i, X_i, V, U, lr=self.lr)
                obj_tr = self.TV(X @ U, Y @ V)
                obj_val = self.TV(X_val @ U, Y_val @ V)
                self.callback(obj_tr, obj_val)
            self.callback(obj_tr, obj_val, epoch=epoch, start_time=start_time)
        return U, V
