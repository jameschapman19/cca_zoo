"""
EigenGame: PCA as a Nash Equilibrium
https://arxiv.org/pdf/2010.00554.pdf
"""
# Importing necessary libraries
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from .utils import TV, initialize


# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V holds the previously computed eigenvectors
@partial(jit, static_argnums=(3))
def model(u, X, U, k):
    M = X.T @ X
    rewards = u.T @ M @ u
    penalties = 0
    for j in range(k):
        penalties = penalties + (u.T @ M @ U[:, j]) ** 2 / (
                U[:, j].T @ M @ U[:, j])
    return jnp.sum(rewards - penalties)


# Update rule to be used for calculating eigenvectors
def update(u, X, U, k, lr=1, riemannian_projection=False):
    du = jax.grad(model)(u, X, U, k)
    if riemannian_projection:
        dur = du - (u.T @ u) * u
        uhat = u + lr * dur
    else:
        uhat = u + lr * du
    return uhat / jnp.linalg.norm(uhat)


def calc_game(X, k, lr=1, iterations=100, riemannian_projection=False, initialization='random',
              random_state=0, simultaneous=False):
    U = initialize(X, k, type=initialization, random_state=random_state)
    obj = []
    if simultaneous:
        for i in range(iterations):
            for k_ in range(k):
                u = update(U[:, k_], X, U, k_, lr=lr, riemannian_projection=riemannian_projection)
                U = U.at[:, k_].set(u)
            obj.append(TV(X, U))
            print(f'iteration {i}: {obj[-1]}')
    else:
        for k_ in range(k):
            for i in range(iterations):
                u = update(U[:, k_], X, U, k_, lr=lr, riemannian_projection=riemannian_projection)
                U = U.at[:, k_].set(u)
                obj.append(TV(X, U))
                print(f'iteration {i}: {obj[-1]}')
    return TV(X, U), U, obj
