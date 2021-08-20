# Importing necessary libraries
import jax.numpy as jnp
from jax import jit

from .utils import initialize, TCC


# Update rule to be used for calculating eigenvectors
@jit
def update(X, Y, Hx, Hy, U, V):
    vhat = jnp.dot(Hy, X @ U)
    uhat = jnp.dot(Hx, Y @ V)
    return jnp.linalg.qr(uhat)[0], jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_lscca_exact(X, Y, k, iterations=100, initialization='uniform',
                     random_state=0):
    U, V = initialize(X, Y, k, initialization, random_state)
    Hx = jnp.linalg.inv(X.T @ X) @ X.T
    Hy = jnp.linalg.inv(Y.T @ Y) @ Y.T
    for i in range(iterations):
        U, V = update(X, Y, Hx, Hy, U, V)
        print(f'iteration {i}: {TCC(X, Y, U, V)}')
    return TCC(X, Y, U, V), U, V
