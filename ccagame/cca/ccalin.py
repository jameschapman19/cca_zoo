# Importing necessary libraries

import jax.numpy as jnp
import jax.scipy as jsp
from jax import random, jit

from ccagame.solver import svrg_solve
from .utils import calc_eigenvalues


@jit
def obj(W, A, B, Wt):
    return jnp.dot(jnp.dot(jnp.transpose(W), B), W) - 2 * jnp.dot(jnp.dot(jnp.transpose(W), A), Wt)


# @partial(jit, static_argnums=(2, 3, 4))
def GenELinK(A, B, n, iterations=100, random_state=0):
    key = random.PRNGKey(random_state)
    d = A.shape[1]
    W = random.normal(key, (d, n))
    W = jnp.linalg.qr(W)[0]
    for i in range(iterations):
        gamma = jnp.dot(jnp.linalg.inv(jnp.dot(jnp.dot(jnp.transpose(W), B), W)),
                        jnp.dot(jnp.dot(jnp.transpose(W), A), W))
        W = svrg_solve(obj, A, B, x=gamma)
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
