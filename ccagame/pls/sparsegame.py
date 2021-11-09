# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import grad, jit

from ccagame.utils import data_stream, get_num_batches
from . import _PLS


def prox(u, t):
    return jnp.sign(u) * jnp.maximum(jnp.abs(u) - t, 0)


@partial(jit, static_argnums=(6, 7))
def alpha_model(u, v, X, Y, U, V, k: int, n: int):
    """
    Returns the utilities for the kth players

    Parameters
    ----------
    u :
        current estimate for this level's left eigenvector
    v :
        current estimate for this level's right eigenvector
    X :
        batch of data for view X
    Y :
        batch of data for view Y
    U :
        all eigenvector estimates for each level
    V :
        all eigenvector estimates for each level
    k :
        level

    Returns
    -------

    """
    C_xy = X.T @ Y
    rewards = u.T @ C_xy @ v
    penalties = (u.T @ C_xy @ V[:, :k]) ** 2 / jnp.diag(U[:, :k].T @ C_xy @ V[:, :k])
    return jnp.sum(rewards - penalties.sum())


@partial(jit, static_argnums=(6))
def mu_model(u, v, X, Y, U, V, k: int):
    """
    Returns the gradients for the kth players

    Parameters
    ----------
    u :
        current estimate for this level's left eigenvector
    v :
        current estimate for this level's right eigenvector
    X :
        batch of data for view X
    Y :
        batch of data for view Y
    U :
        all eigenvector estimates for each level
    V :
        all eigenvector estimates for each level
    k :
        level

    Returns
    -------

    """
    C_xy = X.T @ Y
    rewards = C_xy @ v
    penalties = (u.T @ C_xy @ V[:, :k]) * U[:, :k]
    return rewards - penalties.sum(axis=1)


# Update rule to be used for calculating eigenvectors
@partial(
    jit, static_argnums=(6, 7), static_argnames=("lr", "riemannian_projection", "mu")
)
def update(
    u,
    v,
    X,
    Y,
    U,
    V,
    c,
    k: int,
    n: int,
    lr: float = 1,
    riemannian_projection: bool = False,
    mu=False,
):
    """
    Update the left and right singular vector estimates

    Parameters
    ----------
    u :
        current estimate for this level's left eigenvector
    v :
        current estimate for this level's right eigenvector
    X :
        batch of data for view X
    Y :
        batch of data for view Y
    U :
        all eigenvector estimates for each level
    V :
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
        du = mu_model(u, v, X, Y, U, V, k)
        dv = mu_model(v, u, Y, X, V, U, k)
    else:
        du = grad(alpha_model)(u, v, X, Y, U, V, k)
        dv = grad(alpha_model)(v, u, Y, X, V, U, k)
    if riemannian_projection:
        dur = du - (u.T @ u) * u
        uhat = u + lr * dur
        dvr = dv - (v.T @ v) * v
        vhat = v + lr * dvr
    else:
        uhat = u + lr * du
        vhat = v + lr * dv
    return prox(uhat, lr * c) / jnp.linalg.norm(uhat), prox(
        vhat, lr * c
    ) / jnp.linalg.norm(vhat)


# Object form
class SparseGame(_PLS):
    def __init__(
        self,
        n_components=4,
        *,
        scale=True,
        copy=True,
        lr: float = 1,
        epochs: int = 100,
        riemannian_projection: bool = False,
        random_state: int = 0,
        simultaneous: bool = True,
        batch_size: int = 128,
        mu=True,
        verbose=False,
        wandb=False,
        c=0.0
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
        self.riemannian_projection = riemannian_projection
        self.simultaneous = simultaneous
        self.batch_size = batch_size
        self.mu = mu
        self.c = c

    def _fit(self, X, Y, X_val=None, Y_val=None):
        """

        Parameters
        ----------
        X
        Y

        Returns
        -------

        """
        U, V = self.initialize(X, Y, self.n_components, "random", self.random_state)
        batches = data_stream(X, Y, batch_size=self.batch_size)
        num_batches = get_num_batches(X, Y, batch_size=self.batch_size)
        self.obj = []
        # We can either solve the eigenvectors simulataneously or complete each one
        if self.simultaneous:
            for epoch in range(self.epochs):
                start_time = time.time()
                for b in range(num_batches):
                    _, (X_i, Y_i) = next(batches)
                    for k_ in range(self.n_components):
                        u, v = update(
                            U[:, k_],
                            V[:, k_],
                            X_i,
                            Y_i,
                            U,
                            V,
                            self.c,
                            k_,
                            X.shape[0],
                            lr=self.lr,
                            riemannian_projection=self.riemannian_projection,
                            mu=self.mu,
                        )
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                    obj_tr = self.TV(X @ U, Y @ V)
                    obj_val = self.TV(X_val @ U, Y_val @ V)
                    self.callback(obj_tr, obj_val)
                self.callback(obj_tr, obj_val, epoch=epoch, start_time=start_time)
        else:
            for k_ in range(self.n_components):
                for epoch in range(self.epochs):
                    start_time = time.time()
                    for b in range(num_batches):
                        _, (X_i, Y_i) = next(batches)
                        u, v = update(
                            U[:, k_],
                            V[:, k_],
                            X_i,
                            Y_i,
                            U,
                            V,
                            self.c,
                            k_,
                            X.shape[0],
                            lr=self.lr,
                            riemannian_projection=self.riemannian_projection,
                            mu=self.mu,
                        )
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                        obj_tr = self.SPLS(X, Y, U, V)
                        obj_val = self.SPLS(X_val, Y_val, U, V)
                        self.callback(obj_tr, obj_val)
                    self.callback(obj_tr, obj_val, epoch=epoch, start_time=start_time)
        return U, V

    def SPLS(self, X, Y, U, V):
        X = X @ U
        Y = Y @ V
        C = X.T @ Y
        _, S, _ = jnp.linalg.svd(C)
        reg = self.c * (jnp.linalg.norm(U, 1) + jnp.linalg.norm(V, 1))
        return S.sum() - reg
