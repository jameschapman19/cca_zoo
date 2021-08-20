# Importing necessary libraries

import jax.numpy as jnp

from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
# @partial(jit, static_argnums=(2))
def update(X, Y, U, V):
    U = X.T @ Y @ V
    V = Y.T @ X @ U
    return jnp.linalg.qr(U), jnp.linalg.qr(V)


# Run the update step iteratively across all eigenvectors
def calc_batch(X, Y, k, iterations=100,
               random_state=0):
    U, V = initialize(X, Y, k, 'random', random_state)
    for i in range(iterations):
        U, V = update(X, Y, U, V)
        print(f'iteration {i}: {TV(X, Y, U, V)}')
    return TV(X, Y, U, V), U, V
