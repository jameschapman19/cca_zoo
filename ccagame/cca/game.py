# Importing necessary libraries
import time
from functools import partial

import jax.numpy as jnp
from jax import grad, jit

from ccagame.utils import data_stream, get_num_batches
from . import _CCA
from .utils import initialize, TCC


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
    rewards = C_xt[:, k] - X.T@X@u
    penalties = (u.T @ C_xt[:, :k]) * U[:, :k]
    return rewards - penalties.sum(axis=1)


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=7, static_argnames=("lr", "mu"))
def update(
        u,
        v,
        X,
        Y,
        U,
        V,
        T,
        k: int,
        lr: float = 1,
        mu=True,
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
    mu :
        which game model to use. If True uses unbiased estimate as in eigengame:unloaded if False uses biased estimate as in original eigengame
    """
    if mu:
        du = mu_model(u, X, U, T, k)
        dv = mu_model(v, Y, V, T, k)
    else:
        du = grad(alpha_model)(u, X, U, T, k)
        dv = grad(alpha_model)(v, Y, V, T, k)
    uhat = u + lr * du
    vhat = v + lr * dv
    return uhat, vhat


def update_T(X, Y, U, V):
    latent_dims = U.shape[1]
    W = jnp.eye(latent_dims) - 1
    P, S, Qt = jnp.linalg.svd((X @ U + Y @ V) @ W)
    T = P[:, :latent_dims] @ Qt[:latent_dims, :]
    return T


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
        U, V = initialize(X, Y, self.n_components, "random", self.random_state)
        batches = data_stream(X, Y, batch_size=self.batch_size)
        num_batches = get_num_batches(X, Y, batch_size=self.batch_size)
        if self.simultaneous:
            for epoch in range(self.epochs):
                T = update_T(X, Y, U, V)
                start_time = time.time()
                for _ in range(num_batches):
                    idx, (X_i, Y_i) = next(batches)
                    for k_ in range(self.n_components):
                        u, v = update(
                            U[:, k_],
                            V[:, k_],
                            X_i,
                            Y_i,
                            U,
                            V,
                            T[idx],
                            k_,
                            lr=self.lr,
                            mu=self.mu
                        )
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                epoch_time = time.time() - start_time
                if self.verbose:
                    print(f"Epoch {epoch} in {epoch_time} sec")
                    print(f"epoch {epoch}: {TCC(X, Y, U, V)}")
        else:
            for k_ in range(self.n_components):
                T = update_T(X, Y, U, V)
                for epoch in range(self.epochs):
                    start_time = time.time()
                    for _ in range(num_batches):
                        idx, (X_i, Y_i) = next(batches)
                        u, v = update(
                            U[:, k_],
                            V[:, k_],
                            X_i,
                            Y_i,
                            U,
                            V,
                            T,
                            k_,
                            lr=self.lr,
                            mu=self.mu
                        )
                        U = U.at[:, k_].set(u)
                        V = V.at[:, k_].set(v)
                    epoch_time = time.time() - start_time
                    if self.verbose:
                        print(f"Epoch {epoch} in {epoch_time} sec")
                        print(f"epoch {epoch}: {TCC(X, Y, U, V)}")
        return U, V
