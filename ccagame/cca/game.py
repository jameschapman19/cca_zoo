# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import grad, jit

from ccagame.utils import data_stream, get_num_batches
from . import _CCA


@partial(jit, static_argnums=(4))
def alpha_model(u, X, U, T, k: int):
    """
    Returns the utilities for the kth players

    Parameters
    ----------
    u :
        current estimate for this level's left eigenvector
    X :
        batch of data for view X
    U :
        all eigenvector estimates for each level
    T :
        the shared target
    k :
        level

    Returns
    -------

    """
    C_xt = X.T @ T
    rewards = -jnp.linalg.norm(T[:, k] - X @ u) ** 2
    penalties = (u.T @ C_xt[:, :k]) ** 2 / jnp.diag(U[:, :k].T @ C_xt[:, :k])
    return jnp.sum(rewards - penalties.sum())


@partial(jit, static_argnums=(4))
def mu_model(u, X, U, T, k: int):
    """
    Returns the gradients for the kth players

    Parameters
    ----------
    u :
        current estimate for this level's left eigenvector
    X :
        batch of data for view X
    U :
        all eigenvector estimates for each level
    T :
        the shared target
    k :
        level

    Returns
    -------

    """
    C_xt = X.T @ T
    rewards = C_xt[:, k] - X.T @ X @ u
    penalties = (u.T @ C_xt[:, :k]) * U[:, :k]
    return rewards - penalties.sum(axis=1)


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=6, static_argnames=("lr", "mu"))
def update(
        u,
        v,
        X,
        Y,
        U,
        V,
        k: int,
        lr: float = 1,
        mu=True,
):
    """
    Update the left and right singular vector estimates

    Parameters
    ----------

    """
    T = X @ U + Y @ V
    T = T / jnp.linalg.norm(T, axis=0)
    if mu:
        du = mu_model(u, X, U, T, k)
        dv = mu_model(v, Y, V, T, k)
    else:
        du = grad(alpha_model)(u, X, U, T, k)
        dv = grad(alpha_model)(v, Y, V, T, k)
    du = du * X.shape[0]
    dv = dv * X.shape[0]
    uhat = u + lr * du
    vhat = v + lr * dv
    return uhat, vhat


class Game(_CCA):
    def __init__(
            self,
            n_components=4,
            *,
            scale=True,
            copy=True,
            lr: float = 1.0,
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

    def _fit(self, X, Y, X_val=None, Y_val=None):
        U, V = self.initialize(X, Y, self.n_components, "random", self.random_state)
        batches = data_stream(X, Y, batch_size=self.batch_size)
        num_batches = get_num_batches(X, Y, batch_size=self.batch_size)
        self.obj = []
        if self.simultaneous:
            for epoch in range(self.epochs):
                start_time = time.time()
                for b in range(num_batches):
                    idx, (X_i, Y_i) = next(batches)
                    for k_ in range(self.n_components):
                        u, v = update(
                            U[:, k_],
                            V[:, k_],
                            X_i,
                            Y_i,
                            U,
                            V,
                            k_,
                            X.shape[0],
                            lr=self.lr,
                            mu=self.mu
                        )
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                        obj_tr = self.TCC(X @ U, Y @ V)
                        obj_val = self.TCC(X_val @ U, Y_val @ V)
                        self.callback(obj_tr, obj_val)
                self.callback(obj_tr, obj_val, epoch=epoch, start_time=start_time)
        else:
            for k_ in range(self.n_components):
                for epoch in range(self.epochs):
                    start_time = time.time()
                    for b in range(num_batches):
                        idx, (X_i, Y_i) = next(batches)
                        u, v = update(
                            U[:, k_],
                            V[:, k_],
                            X_i,
                            Y_i,
                            U,
                            V,
                            k_,
                            X.shape[0],
                            lr=self.lr,
                            mu=self.mu
                        )
                        # T,
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                        obj_tr = self.TCC(X @ U, Y @ V)
                        obj_val = self.TCC(X_val @ U, Y_val @ V)
                        self.callback(obj_tr, obj_val)
                    self.callback(obj_tr, obj_val, epoch=epoch, start_time=start_time)
        return U, V
