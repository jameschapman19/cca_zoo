# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit, grad

from .utils import initialize, calc_eigenvalues, TCC


# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V holds the previously computed eigenvectors
@partial(jit, static_argnums=5)
def model(u, v, X, Y, U, k):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    rewards = u.T@C_xy@v
    penalties = 0
    for j in range(k):
        penalties = penalties + u.T@U[:, j]
    return jnp.sum(rewards - penalties)


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=6, static_argnames=('lr', 'riemannian_projection'))
def update(u, v, X, Y, U, V, k, lr=1, riemannian_projection=False):
    du = grad(model)(u, v, X, Y, U, k)
    dv = grad(model)(v, u, Y, X, V, k)
    if riemannian_projection:
        dur = du - (u.T@u) * u
        uhat = u + lr * dur
        dvr = dv - (v.T@v) * v
        vhat = v + lr * dvr
    else:
        uhat = u + lr * du
        vhat = v + lr * dv
    return uhat / jnp.linalg.norm(uhat), vhat / jnp.linalg.norm(vhat)


# Run the update step iteratively across all eigenvectors
def calc_game(X, Y, k, lr=1, iterations=100, riemannian_projection=False, initialization='uniform',
              random_state=0, simultaneous=False):
    U, V = initialize(X, Y, k, initialization, random_state)
    if simultaneous:
        for i in range(iterations):
            for k_ in range(k):
                u, v = update(U[:, k_], V[:, k_], X, Y, U, V, k_, lr=lr, riemannian_projection=riemannian_projection)
                U = U.at[:, k_].set(u)
                V = V.at[:, k_].set(v)
            print(f'iteration {i}: {TCC(X, Y, U, V)}')
    else:
        for k_ in range(k):
            for i in range(iterations):
                u, v = update(U[:, k_], V[:, k_], X, Y, U, V, k_, lr=lr, riemannian_projection=riemannian_projection)
                U = U.at[:, k_].set(u)
                V = V.at[:, k_].set(v)
                print(f'iteration {i}: {TCC(X, Y, U, V)}')
    return calc_eigenvalues(X, Y, U, V), U, V
