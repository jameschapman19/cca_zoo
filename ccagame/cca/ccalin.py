# Importing necessary libraries
from functools import partial
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random, jit

from ccagame.solver import agd_solve
from .utils import calc_eigenvalues


@jit
def obj(W, A, B, Wt):
    return jnp.trace(jnp.dot(jnp.dot(jnp.transpose(W), B), W) - 2 * jnp.dot(jnp.dot(jnp.transpose(W), A), Wt))


@jit
def gamma(W, A, B):
    return jnp.dot(jnp.linalg.inv(jnp.dot(jnp.dot(jnp.transpose(W), B), W)),
                   jnp.dot(jnp.dot(jnp.transpose(W), A), W))


@jit
def GenELinK_update(W, A, B):
    W = agd_solve(obj, A, B, W, x=jnp.dot(W, gamma(W, A, B)))
    return jnp.linalg.qr(W)[0]


def GenELinK(A, B, k, iterations=100, random_state=0):
    key = random.PRNGKey(random_state)
    d = A.shape[1]
    W = random.normal(key, (d, k))
    W = jnp.linalg.qr(W)[0]
    for i in range(iterations):
        W = GenELinK_update(W, A, B)
    return W


# Run the update step iteratively across all eigenvectors
def calc_ccalin(X, Y, k, iterations=5, random_state=0):
    A = jnp.hstack((X, Y))
    A = jnp.dot(jnp.transpose(A), A)
    B = jsp.linalg.block_diag(jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y))
    A = A - B
    W = GenELinK(A, B, k, iterations=iterations, random_state=random_state)
    key = random.PRNGKey(random_state)
    U = random.normal(key, (2 * k, k))
    Wx = jnp.linalg.qr(jnp.dot(W[:A.shape[1]], U))
    Wy = jnp.linalg.qr(jnp.dot(W[A.shape[1]:], U))
    return calc_eigenvalues(X, Y, Wx, Wy), Wx, Wy
