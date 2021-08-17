# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit, grad

from .utils import initialize, calc_eigenvalues


# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V holds the previously computed eigenvectors
@partial(jit, static_argnums=(5))
def model(u, v, X, Y, V, k):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    C_xx = jnp.dot(jnp.transpose(X), X)
    C_yy = jnp.dot(jnp.transpose(Y), Y)
    rewards = jnp.dot(jnp.transpose(u), jnp.dot(C_xy, v)) ** 2 / (
            jnp.dot(jnp.transpose(u), jnp.dot(C_xx, u)) * jnp.dot(jnp.transpose(v), jnp.dot(C_yy, v)))
    penalties = 0
    for j in range(k):
        penalties = penalties + jnp.dot(jnp.transpose(u), jnp.dot(C_xy, V[:, j].reshape(-1, 1))) ** 2 / (jnp.dot(
            jnp.transpose(V[:, j].reshape(-1, 1)), jnp.dot(C_yy, V[:, j].reshape(-1, 1))) * jnp.dot(
            jnp.transpose(u), jnp.dot(C_xx, u)))
    return jnp.sum(rewards - penalties)


# Update rule to be used for calculating eigenvectors
def update(u, v, X, Y, U, V, k, lr=1e-1, riemannian_projection=False):
    du = grad(model)(u, v, X, Y, V, k)
    dv = grad(model)(v, u, Y, X, U, k)
    if riemannian_projection:
        dur = du - (jnp.dot(du.T, u)) * u
        uhat = u + lr * dur
        dvr = dv - (jnp.dot(dv.T, v)) * v
        vhat = v + lr * dvr
    else:
        uhat = u + lr * du
        vhat = v + lr * dv
    return uhat / jnp.linalg.norm(uhat), vhat / jnp.linalg.norm(vhat)


# Run the update step iteratively across all eigenvectors
def calc_game(X, Y, k, lr=1e-1, iterations=100, riemannian_projection=False, initialization='uniform',
              random_state=0, simultaneous=False):
    U, V = initialize(X, Y, k, initialization, random_state)
    if simultaneous:
        for i in range(iterations):
            for k_ in range(k):
                u, v = update(U[:, k], V[:, k], X, Y, U, V, k_, lr=lr, riemannian_projection=riemannian_projection)
                U = U.at[:, k_].set(u)
                V = V.at[:, k_].set(v)
            print(f'iteration {i}: {calc_eigenvalues(X, Y, U, V)}')
    else:
        for k_ in range(k):
            for i in range(iterations):
                u, v = update(U[:, k_], V[:, k_], X, Y, U, V, k_, lr=lr, riemannian_projection=riemannian_projection)
                U = U.at[:, k_].set(u)
                V = V.at[:, k_].set(v)
                print(f'iteration {i}: {calc_eigenvalues(X, Y, U, V)}')
    return calc_eigenvalues(X, Y, U, V), U, V
