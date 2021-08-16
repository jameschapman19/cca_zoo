# Importing necessary libraries
import jax.numpy as jnp
from jax import jit

from .utils import calc_eigenvalues


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
@jit
def update(X, Y, U1, V1):
    vhat = jnp.linalg.lstsq(Y, jnp.dot(X, U1))[0]
    uhat = jnp.linalg.lstsq(X, jnp.dot(Y, V1))[0]
    return jnp.linalg.qr(uhat)[0], jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_lscca(X, Y, n, iterations=100, initialization='uniform'):
    X_aux = X[:n_aux]
    Y_aux = Y[:n_aux]
    X = X[n_aux:]
    Y = Y[n_aux:]
    Cxx = jnp.dot(jnp.transpose(X_aux), X_aux) / n_aux
    Cyy = jnp.dot(jnp.transpose(Y_aux), Y_aux) / n_aux
    for i in range(iterations):
        U1, V1 = update(X, Y, U1, V1)
        print(f'iteration {i}: {calc_eigenvalues(X, Y, U1, V1)}')
    return calc_eigenvalues(X, Y, U1, V1), U1, V1
