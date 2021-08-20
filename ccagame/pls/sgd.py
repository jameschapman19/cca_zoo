# Importing necessary libraries

import jax.numpy as jnp

from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
# @partial(jit, static_argnums=(2))
def update(X, Y, U, V, lr=0.1):
    du = X.T @ Y @ V
    uhat = U + lr * du
    return jnp.linalg.qr(uhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_sgd(X, Y, k, lr=1, iterations=100,
             random_state=0):
    U, V = initialize(X, Y, k, 'random', random_state)
    for i in range(iterations):
        U = update(X, Y, U, V, lr=lr)
        V = update(Y, X, V, U, lr=lr)
        print(f'iteration {i}: {TV(X, Y, U, V)}')
    return TV(X, Y, U, V), U, V
