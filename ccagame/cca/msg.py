# TODO
# https://export.arxiv.org/pdf/1702.06818

# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
import wandb
from jax import jit
from sklearn.model_selection import train_test_split

from ccagame.utils import data_stream, get_num_batches
from . import _CCA
from .utils import TCC, initialize


def invsqrtm(M):
    U, S, Vt = jnp.linalg.svd(M)
    S = S ** -0.5
    return U @ jnp.diag(S) @ Vt


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(6, 7), static_argnames=('lr'))
def update(X, Y, U, V, Cx, Cy, b, k, lr: float = 0.1):
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
    b = b + 1
    Cx = (b - 1) * Cx / b + X.T @ X / b
    Cy = (b - 1) * Cy / b + Y.T @ Y / b
    Wx = invsqrtm(Cx)
    Wy = invsqrtm(Cy)
    delta = Wx.T @ X.T @ Y @ Wy
    M = U @ V.T + lr * delta
    U, _, Vt = jnp.linalg.svd(M)
    return U[:, :k], Vt[:k].T, Cx, Cy


# Object form
class MSG(_CCA):
    def __init__(
            self,
            n_components=2,
            *,
            scale=True,
            copy=True,
            lr: float = 1,
            epochs: int = 100,
            random_state: int = 0,
            batch_size: int = 128,
            verbose=False,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def _fit(self, X, Y):
        X, X_val, Y, Y_val = train_test_split(
            X, Y, random_state=self.random_state, train_size=0.9
        )
        U, V = initialize(X, Y, self.n_components, "random", self.random_state)
        Cx = jnp.zeros((X.shape[1], X.shape[1]))
        Cy = jnp.zeros((Y.shape[1], Y.shape[1]))
        batches = data_stream(X, Y, batch_size=self.batch_size)
        num_batches = get_num_batches(X, Y, batch_size=self.batch_size)
        self.obj = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for b in range(num_batches):
                _, (X_i, Y_i) = next(batches)
                U, V, Cx, Cy = update(X_i, Y_i, U, V, Cx, Cy, b, self.n_components, lr=self.lr)
                obj = TCC(X, Y, U, V)
                if self.wandb:
                    wandb.log({"Iteration/Objective": obj}, step=b)
                else:
                    self.obj.append(obj)
            obj = TCC(X, Y, U, V)
            if self.wandb:
                wandb.log({"Epoch/Objective": obj}, step=epoch)
            if self.verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch} in {epoch_time} sec")
                print(f"epoch {epoch} objective: {obj}")
        return U, V
