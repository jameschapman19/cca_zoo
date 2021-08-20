"""
Efficient Globally Convergent Stochastic
Optimization for Canonical Correlation Analysis
https://proceedings.neurips.cc/paper/2016/file/42998cf32d552343bc8e460416382dca-Paper.pdf
"""
# Importing necessary libraries
import jax.numpy as jnp
from jax import jit

from .utils import initialize, calc_eigenvalues, TCC


# Update rule to be used for calculating eigenvectors
@jit
def update(X, Y, U, V):
    vhat = jnp.linalg.lstsq(Y, X@U)[0]
    uhat = jnp.linalg.lstsq(X, Y@V)[0]
    return jnp.linalg.qr(uhat)[0], jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_lscca(X, Y, k, iterations=100, initialization='uniform',
               random_state=0):
    U, V = initialize(X, Y, k, initialization, random_state)
    for i in range(iterations):
        U, V = update(X, Y, U, V)
        print(f'iteration {i}: {TCC(X, Y, U, V)}')
    return TCC(X, Y, U, V), U, V
