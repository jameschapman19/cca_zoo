# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.optimize
from jax import jit
from jax import random

from .utils import calc_eigenvalues


def obj(W, A, B, Wt):
    return jnp.dot(jnp.dot(jnp.transpose(W), B), W) - 2 * jnp.dot(jnp.dot(jnp.transpose(W), A), Wt)


def SVRG(U, V, X, Y, rx, T, m, eta):
    for t in range(T):
        W_ = U
        W = W_
        batch_grad = jnp.dot(X, jnp.dot(jnp.transpose(X), jnp.dot(W_ - Y, V)))+rx*W_
    return y


@partial(jit, static_argnums=(2, 3, 4))
def GenELinK(A, B, n, iterations=100, random_state=0):
    key = random.PRNGKey(random_state)
    d = A.shape[1]
    W = random.normal(key, (d, n))
    W = jnp.linalg.qr(W)[0]
    for i in range(iterations):
        gamma = jnp.dot(jnp.linalg.inv(jnp.dot(jnp.dot(jnp.transpose(W), B), W)),
                        jnp.dot(jnp.dot(jnp.transpose(W), A), W))

        W = jnp.linalg.qr(W)[0]
    return W


# Run the update step iteratively across all eigenvectors
def calc_ccalin(X, Y, n, iterations=100, random_state=0):
    A = jnp.hstack((X, Y))
    A = jnp.dot(jnp.transpose(A), A)
    B = jsp.linalg.block_diag(jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y))
    A = A - B
    W = GenELinK(A, B, n, iterations=iterations, random_state=random_state)
    key = random.PRNGKey(random_state)
    U = random.normal(key, (2 * n, n))
    Wx = jnp.linalg.qr(jnp.dot(W[:A.shape[1]], U))
    Wy = jnp.linalg.qr(jnp.dot(W[A.shape[1]:], U))
    return calc_eigenvalues(X, Y, Wx, Wy), Wx, Wy
