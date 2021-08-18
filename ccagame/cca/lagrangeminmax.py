"""
On Constrained Nonconvex Stochastic Optimization: A Case
Study for Generalized Eigenvalue Decomposition
http://proceedings.mlr.press/v89/chen19a/chen19a.pdf
"""
# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit

from ccagame.cca.utils import initialize


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(5))
def update(A, B, W, Y, lr):
    W = W - lr * (jnp.dot(jnp.dot(B, W), Y) - jnp.dot(A, W))
    Y = jnp.dot(jnp.transpose(W), jnp.dot(A, W))
    return W, Y


# Run the update step iteratively across all eigenvectors
def calc_lagrangeminmax(X, Y, k, iterations=100,
                        lr=10, random_state=0, initialization='random'):
    n = X.shape[0]
    p = X.shape[1]
    A = jnp.hstack((X, Y))
    A = jnp.dot(jnp.transpose(A), A) / n
    B = jsp.linalg.block_diag(jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y)) / n
    A = A - B
    W, V = initialize(X, Y, k, type=initialization, random_state=random_state)
    W = jnp.vstack((W, V))
    V = jnp.zeros((k, k))
    for i in range(iterations):
        W, V = update(A, B, W, V, lr)
        Wx = W[:p]
        Wy = W[p:]
        print(f'iteration {i}: {jnp.sum(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy)))}')
    return jnp.sum(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy))), Wx, Wy
