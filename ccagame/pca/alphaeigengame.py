# Importing necessary libraries

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import random

from .eigengamevals import calc_eigengame_eigenvalues


# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V1 holds the previously computed eigenvectors
@jit
def model(v, X, V1):
    M = jnp.dot(jnp.transpose(X), X)
    rewards = jnp.dot(jnp.transpose(v), jnp.dot(M, v))
    penalties = 0
    for j in range(np.size(V1[:, V1.any(0)], axis=1) - 1):
        penalties = penalties + (jnp.dot(jnp.transpose(v), jnp.dot(M, V1[:, j].reshape(-1, 1)))) ** 2 / jnp.dot(
            jnp.transpose(V1[:, j].reshape(-1, 1)), jnp.dot(M, V1[:, j].reshape(-1, 1)))
    return jnp.sum(rewards - penalties)


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
def update(v, X, V1, lr=0.1, riemannian_projection=False):
    dv = jax.grad(model)(v, X, V1)
    if riemannian_projection:
        dvr = dv - (jnp.dot(dv.T, v)) * v
        vhat = v + lr * dvr
    else:
        vhat = v + lr * dv
    return (vhat / jnp.linalg.norm(vhat))


# Run the update step iteratively across all eigenvectors
def calc_alphaeigengame(X, n, lr=1e-1, iterations=100, riemannian_projection=False, initialization='random',
                        random_state=0, simultaneous=False):
    if initialization == 'uniform':
        V1 = jnp.ones((n, 1))
        V1 = V1 / jnp.linalg.norm(V1)
    elif initialization == 'random':
        key = random.PRNGKey(random_state)
        V1 = random.normal(key, (X.shape[1], n))
        V1 = V1 / jnp.linalg.norm(V1)
    else:
        print(f'Initialization "{initialization}" not implemented')
        return

    if simultaneous:
        for k in range(n):
            print("Finding the eigenvector ", k)
            for i in range(iterations):
                v = update(V1[:, k], X, V1, lr=lr, riemannian_projection=riemannian_projection)
                V1 = V1.at[:, k].set(v)
                print(f'iteration {i}: {calc_eigengame_eigenvalues(X, V1)}')
    else:
        for i in range(iterations):
            for k in range(n):
                v = update(V1[:, k], X, V1, lr=lr, riemannian_projection=riemannian_projection)
                V1 = V1.at[:, k].set(v)
                print(f'iteration {i}: {calc_eigengame_eigenvalues(X, V1)}')
    return calc_eigengame_eigenvalues(X, V1), V1
