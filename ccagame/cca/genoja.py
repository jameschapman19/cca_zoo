# Importing necessary libraries
from functools import partial

import jax.scipy as jsp
import jax.numpy as jnp
from jax import jit, grad

from .utils import initialize, calc_eigenvalues


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
#@partial(jit, static_argnums=(4, 5))
def update(X, Y, W, V, alpha=1e-1, beta=1e-1):
    A = jnp.hstack((X, Y))
    A = jnp.dot(jnp.transpose(A), A)
    B = jsp.linalg.block_diag((jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y)))
    W = W - alpha * (jnp.dot(B, W) - jnp.dot(A, V))
    V = V + beta * W
    return W,jnp.linalg.qr(V)[0]


# Run the update step iteratively across all eigenvectors
def calc_genoja(X, Y, n, iterations=100, initialization='uniform',
               random_state=0, alpha=1e-1, beta=1e-1):
    W1, V1 = initialize(X, Y, n, initialization, random_state)
    for i in range(iterations):
        W1, V1 = update(X, Y, W1, V1, alpha=alpha, beta=beta)
        print(f'iteration {i}: {calc_eigenvalues(X, Y, jnp.linalg.qr(V1[:X.shape[1]])[0], jnp.linalg.qr(V1[:Y.shape[1]])[0])}')
    return calc_eigenvalues(X, Y, jnp.linalg.qr(V1[:X.shape[1]])[0], jnp.linalg.qr(V1[:Y.shape[1]])[0]), jnp.linalg.qr(V1[:X.shape[1]])[0], jnp.linalg.qr(V1[:Y.shape[1]])[0]
