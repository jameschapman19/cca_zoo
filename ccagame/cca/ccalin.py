# Importing necessary libraries
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.optimize
from jax import random

from .utils import calc_eigenvalues


def obj(W, A, B):
    return jnp.trace(jnp.dot(jnp.dot(jnp.transpose(W), B), W) - 2 * jnp.dot(jnp.dot(jnp.transpose(W), A), W))


def GenELinK(A, B, n, iterations=100, random_state=0):
    key = random.PRNGKey(random_state)
    d = A.shape[1]
    W = random.normal(key, (d, n))
    W = jnp.linalg.qr(W)[0]
    for i in range(iterations):
        gamma = jnp.dot(jnp.linalg.inv(jnp.dot(jnp.dot(jnp.transpose(W), B), W)),
                        jnp.dot(jnp.dot(jnp.transpose(W), A), W))
        W = jax.scipy.optimize.minimize(obj, jnp.dot(W, gamma), args=(A, B), method='BFGS')
        W = jnp.linalg.qr(W)
    return W


# Run the update step iteratively across all eigenvectors
def calc_ccalin(X, Y, n, iterations=100, initialization='uniform', random_state=0):
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
