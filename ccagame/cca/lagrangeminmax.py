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
                        lr=1e-3, random_state=0, initialization='uniform'):
    n = X.shape[0]
    p = X.shape[1]
    A = jnp.hstack((X, Y))
    A = jnp.dot(jnp.transpose(A), A) / n
    B = jsp.linalg.block_diag(jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y)) / n
    A = A - B
    W, V = initialize(X, Y, k, type=initialization, random_state=random_state)
    W = jnp.vstack(W, V)
    V = W
    for i in range(iterations):
        W, V = update(A, B, W, V, lr)
        Wx = jnp.linalg.qr(W[:p])
        Wy = jnp.linalg.qr(W[p:])
        print(f'iteration {i}: {jnp.sum(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy)))}')
    return jnp.sum(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy))), Wx, Wy
