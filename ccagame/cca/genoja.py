# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import random

from ccagame.cca.utils import calc_eigenvalues


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
@partial(jit, static_argnums=(7))
def update(X, Y, wx, wy, vx, vy, beta, alpha=1e-3):
    Cxx = jnp.dot(jnp.transpose(X), X)
    Cyy = jnp.dot(jnp.transpose(Y), Y)
    Cxy = jnp.dot(jnp.transpose(X), Y)
    wx = wx - alpha * (jnp.dot(Cxx, wx) - jnp.dot(Cxy, vy))
    wy = wy - alpha * (jnp.dot(Cyy, wy) - jnp.dot(jnp.transpose(Cxy), vx))
    vx = vx + beta * wx
    vy = vy + beta * wy
    v = jnp.hstack((vx, vy))
    vx = vx / jnp.linalg.norm(v)
    vy = vy / jnp.linalg.norm(v)
    return wx, wy, vx, vy


def initialize(X, Y, n, type='uniform', random_state=0):
    if type == 'uniform':
        wx = jnp.ones((X.shape[1], n))
        vx = jnp.ones((X.shape[1], n))
        wx = wx / jnp.linalg.norm(wx, axis=0)
        vx = vx / jnp.linalg.norm(vx, axis=0)
        wy = jnp.ones((Y.shape[1], n))
        vy = jnp.ones((Y.shape[1], n))
        wy = wy / jnp.linalg.norm(wy, axis=0)
        vy = vy / jnp.linalg.norm(vy, axis=0)
    elif type == 'random':
        key = random.PRNGKey(random_state)
        key, subkey = random.split(key)
        wx = random.normal(key, (X.shape[1], n))
        wy = random.normal(subkey, (Y.shape[1], n))
        wx = wx / jnp.linalg.norm(wx, axis=0)
        wy = wy / jnp.linalg.norm(wy, axis=0)
        vx = random.normal(key, (X.shape[1], n))
        vy = random.normal(subkey, (Y.shape[1], n))
        vx = vx / jnp.linalg.norm(vx, axis=0)
        vy = vy / jnp.linalg.norm(vy, axis=0)
    else:
        print(f'Initialization "{type}" not implemented')
        return
    return wx, wy, vx, vy


# Run the update step iteratively across all eigenvectors
def calc_genoja(X, Y, n, iterations=100, initialization='uniform',
                random_state=0, alpha=1e-4, beta_0=1e-3):
    wx, wy, vx, vy = initialize(X, Y, n, initialization, random_state)
    for i in range(iterations):
        wx, wy, vx, vy = update(X, Y, wx, wy, vx, vy, beta=beta_0 / (1 + 1e-4 * i), alpha=alpha)
        print(
            f'iteration {i}: {calc_eigenvalues(X, Y, vx, vy)}')
    return calc_eigenvalues(X, Y, vx, vy), vx, vy
