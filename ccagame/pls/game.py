# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from jax import grad, jit
import wandb
from ccagame.utils import data_stream, get_num_batches
from . import _PLS
from .utils import initialize, TV


@partial(jit, static_argnums=(6))
def alpha_model(u, v, X, Y, U, V, k: int):
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
@partial(jit, static_argnums=6, static_argnames=("lr", "riemannian_projection", "mu"))
def update(
    u,
    v,
    X,
    Y,
    U,
    V,
    k: int,
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
    return uhat / jnp.linalg.norm(uhat), vhat / jnp.linalg.norm(vhat)


# Object form
class Game(_PLS):
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
        wandb=False
    ):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb)
        self.lr = lr
        self.epochs = epochs
        self.riemannian_projection = riemannian_projection
        self.random_state = random_state
        self.simultaneous = simultaneous
        self.batch_size = batch_size
        self.mu = mu
        self.verbose = verbose

    def _fit(self, X, Y):
        """

        Parameters
        ----------
        X
        Y

        Returns
        -------

        """
        X, X_val, Y, Y_val = train_test_split(
            X, Y, random_state=self.random_state, train_size=0.9
        )
        U, V = initialize(X, Y, self.n_components, "random", self.random_state)
        batches = data_stream(X, Y, batch_size=self.batch_size)
        num_batches = get_num_batches(X, Y, batch_size=self.batch_size)
        self.obj = []
        # We can either solve the eigenvectors simulataneously or complete each one
        if self.simultaneous:
            for epoch in range(self.epochs):
                start_time = time.time()
                for _ in range(num_batches):
                    X_i, Y_i = next(batches)
                    for k_ in range(self.n_components):
                        u, v = update(
                            U[:, k_],
                            V[:, k_],
                            X_i,
                            Y_i,
                            U,
                            V,
                            k_,
                            lr=self.lr,
                            riemannian_projection=self.riemannian_projection,
                            mu=self.mu,
                        )
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                    obj = TV(X, Y, U, V)
                    if self.wandb:
                        wandb.log({"Iteration/Objective": obj})
                    else:
                        self.obj.append(obj)
                obj = TV(X, Y, U, V)
                if self.wandb:
                    wandb.log({"Epoch/Objective": obj})
                if self.verbose:
                    epoch_time = time.time() - start_time
                    print(f"Epoch {epoch} in {epoch_time} sec")
                    print(f"epoch {epoch} objective: {obj}")
        else:
            for k_ in range(self.n_components):
                for epoch in range(self.epochs):
                    start_time = time.time()
                    for _ in range(num_batches):
                        X_i, Y_i = next(batches)
                        u, v = update(
                            U[:, k_],
                            V[:, k_],
                            X_i,
                            Y_i,
                            U,
                            V,
                            k_,
                            lr=self.lr,
                            riemannian_projection=self.riemannian_projection,
                            mu=self.mu,
                        )
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                        obj = TV(X, Y, U, V)
                        if self.wandb:
                            wandb.log({f"Iteration/Objective/{k_}": obj})
                        else:
                            self.obj.append(obj)
                    obj = TV(X, Y, U, V)
                    if self.wandb:
                        wandb.log({f"Epoch/Objective/{k_}": obj})
                    if self.verbose:
                        epoch_time = time.time() - start_time
                        print(f"Epoch {epoch} in {epoch_time} sec")
                        print(f"epoch {epoch} objective: {obj}")
        return U, V


# Function form
def calc_game(
    X,
    Y,
    k: int,
    lr: float = 1,
    epochs: int = 100,
    riemannian_projection: bool = False,
    random_state: int = 0,
    simultaneous: bool = False,
    batch_size: int = 128,
    mu=True,
):
    """
    Calculate partial least squares weights with PLS-Game

    Parameters
    ----------
    X :
        First view of data
    Y :
        Second view of data
    k :
        number of latent dimensions
    lr :
        learning rate
    epochs :
        number of epochs
    riemannian_projection :
        whether to do a riemannian gradient descent projection. False gives a smoothing effect near the optimum
    random_state :
        random seed
    simultaneous :
        whether to solve for all players simultaneously
    batch_size :
        minibatch size for calculation of stochastic gradients

    Returns
    -------

    """
    U, V = initialize(X, Y, k, "random", random_state)
    batches = data_stream(X, Y, batch_size=batch_size)
    num_batches = get_num_batches(X, Y, batch_size=batch_size)
    # We can either solve the eigenvectors simulataneously or complete each one
    if simultaneous:
        for epoch in range(epochs):
            for _ in range(num_batches):
                X_i, Y_i = next(batches)
                for k_ in range(k):
                    u, v = update(
                        U[:, k_],
                        V[:, k_],
                        X_i,
                        Y_i,
                        U,
                        V,
                        k_,
                        lr=lr,
                        riemannian_projection=riemannian_projection,
                        mu=mu,
                    )
                    U = U.at[:, k_].set(u)
                    V = V.at[:, k_].set(v)
    else:
        for k_ in range(k):
            for epoch in range(epochs):
                for _ in range(num_batches):
                    X_i, Y_i = next(batches)
                    u, v = update(
                        U[:, k_],
                        V[:, k_],
                        X_i,
                        Y_i,
                        U,
                        V,
                        k_,
                        lr=lr,
                        riemannian_projection=riemannian_projection,
                        mu=mu,
                    )
                    U = U.at[:, k_].set(u)
                    V = V.at[:, k_].set(v)
    return TV(X, Y, U, V), U, V
