"""
EigenGame: PCA as a Nash Equilibrium
https://arxiv.org/pdf/2010.00554.pdf
"""
import time
# Importing necessary libraries
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from . import _PCA
# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V holds the previously computed eigenvectors
from ..utils import data_stream, get_num_batches


@partial(jit, static_argnums=3)
def alpha_model(u, X, U, k):
    """
    Returns the utility of the kth player

    Parameters
    ----------
    u
    X
    U
    k

    Returns
    -------

    """
    M = X.T @ X
    rewards = u.T @ M @ u
    penalties = (u.T @ M @ U[:, :k]) ** 2 / jnp.diag(U[:, :k].T @ M @ U[:, :k])
    return jnp.sum(rewards - penalties.sum())


@partial(jit, static_argnums=3)
def mu_model(u, X, U, k):
    """
    Returns the utility of the kth player

    Parameters
    ----------
    u
    X
    U
    k

    Returns
    -------

    """
    M = X.T @ X
    rewards = M @ u
    penalties = u.T @ M @ U[:, :k] * U[:, :k]
    return rewards - penalties.sum(axis=1)


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=3, static_argnames=("lr", "riemannian_projection", "mu"))
def update(u, X, U, k, lr: float = 1.0, riemannian_projection=False, mu=False):
    """
    Update the singular vector estimates
    Parameters
    ----------
    u :
        current estimate for this level's left eigenvector
    X :
        batch of data for view X
    U :
        all eigenvector estimates for each level
    k :
        level
    lr :
        learning rate
    riemannian_projection :
        whether to use riemannian projection
    mu :
        which game model to use. If True uses unbiased estimate as in eigengame:unloaded if False uses biased estimate as in original eigengame
    """
    if mu:
        du = mu_model(u, X, U, k)
    else:
        du = jax.grad(alpha_model)(u, X, U, k)
    if riemannian_projection:
        dur = du - (u.T @ u) * u
        uhat = u + lr * dur
    else:
        uhat = u + lr * du
    return uhat / jnp.linalg.norm(uhat)


# Object form
class Game(_PCA):
    def __init__(
            self,
            n_components=4,
            *,
            scale=True,
            copy=True,
            lr: float = 1,
            epochs: int = 100,
            riemannian_projection: bool = False,
            random_state: int = None,
            simultaneous: bool = True,
            batch_size: int = 128,
            mu=True,
            verbose=False,
            wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)
        self.lr = lr
        self.epochs = epochs
        self.riemannian_projection = riemannian_projection
        self.simultaneous = simultaneous
        self.batch_size = batch_size
        self.mu = mu

    def _fit(self, X, X_val):
        U = self.initialize(
            X, self.n_components, type="random", random_state=self.random_state
        )
        batches = data_stream(X, batch_size=self.batch_size)
        num_batches = get_num_batches(X, batch_size=self.batch_size)
        self.obj = []
        if self.simultaneous:
            for epoch in range(self.epochs):
                start_time = time.time()
                for b in range(num_batches):
                    _, X_i = next(batches)
                    for k_ in range(self.n_components):
                        u = update(
                            U[:, k_],
                            X_i,
                            U,
                            k_,
                            lr=self.lr,
                            riemannian_projection=self.riemannian_projection,
                            mu=self.mu,
                        )
                        U = U.at[:, k_].set(u)
                        obj_tr = self.TV(X @ U)
                        obj_val = self.TV(X_val @ U)
                    self.callback(obj_tr, obj_val, b)
                self.callback(obj_tr, obj_val, epoch, start_time)
        else:
            for k_ in range(self.n_components):
                for epoch in range(self.epochs):
                    start_time = time.time()
                    for b in range(num_batches):
                        _, X_i = next(batches)
                        u = update(
                            U[:, k_],
                            X_i,
                            U,
                            k_,
                            lr=self.lr,
                            riemannian_projection=self.riemannian_projection,
                            mu=self.mu,
                        )
                        U = U.at[:, k_].set(u)
                        obj_tr = self.TV(X @ U)
                        obj_val = self.TV(X_val @ U)
                        self.callback(obj_tr, obj_val, b)
                    self.callback(obj_tr, obj_val, epoch, start_time)
        return U
