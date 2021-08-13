# Importing necessary libraries

import jax.numpy as jnp
from jax import random

from .utils import calc_eigenvalues


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
def update(v, X, lr=0.1):
    dv = jnp.dot(jnp.dot(jnp.transpose(X), X), v) - jnp.dot(jnp.tril(jnp.dot(X, v)),dv)
    vhat = v + lr * dv
    return vhat


# Run the update step iteratively across all eigenvectors
def calc_gha(X, n, lr=1e-1, iterations=100, initialization='random',
             random_state=0):
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

    for i in range(iterations):
        V1 = update(V1, X, lr=lr)
        print(f'iteration {i}: {calc_eigenvalues(X, V1)}')
    return calc_eigenvalues(X, V1), V1