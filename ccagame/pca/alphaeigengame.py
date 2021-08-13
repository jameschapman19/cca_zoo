# Importing necessary libraries
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax import random

from .utils import calc_eigenvalues, initialize


# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V holds the previously computed eigenvectors
@partial(jit, static_argnums=(3))
def model(u, X, U, k):
    M = jnp.dot(jnp.transpose(X), X)
    rewards = jnp.dot(jnp.transpose(u), jnp.dot(M, u))
    penalties = 0
    for j in range(k):
        penalties = penalties + (jnp.dot(jnp.transpose(u), jnp.dot(M, U[:, j].reshape(-1, 1)))) ** 2 / jnp.dot(
            jnp.transpose(U[:, j].reshape(-1, 1)), jnp.dot(M, U[:, j].reshape(-1, 1)))
    return jnp.sum(rewards - penalties)


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
def update(v, X, V, k, lr=0.1, riemannian_projection=False):
    dv = jax.grad(model)(v, X, V, k)
    if riemannian_projection:
        dvr = dv - (jnp.dot(dv.T, v)) * v
        vhat = v + lr * dvr
    else:
        vhat = v + lr * dv
    return (vhat / jnp.linalg.norm(vhat))


# Run the update step iteratively across all eigenvectors
def calc_alphaeigengame(X, n, lr=1e-1, iterations=100, riemannian_projection=False, initialization='random',
                        random_state=0, simultaneous=False):
    V = initialize(X, n, type=initialization, random_state=random_state)
    if simultaneous:
        for i in range(iterations):
            for k in range(n):
                v = update(V[:, k], X, V, k, lr=lr, riemannian_projection=riemannian_projection)
                V = V.at[:, k].set(v)
            print(f'iteration {i}: {calc_eigenvalues(X, V)}')
    else:
        for k in range(n):
            for i in range(iterations):
                v = update(V[:, k], X, V, k, lr=lr, riemannian_projection=riemannian_projection)
                V = V.at[:, k].set(v)
                print(f'iteration {i}: {calc_eigenvalues(X, V)}')
    return calc_eigenvalues(X, V), V
