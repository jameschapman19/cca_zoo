# Importing necessary libraries
from functools import partial
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random, jit

from ccagame.solver import agd_solve, gd_solve
from .utils import calc_eigenvalues, gram_schmidt_matrix


@jit
def obj(W, A, B, Wt):
    return jnp.trace(0.5 * jnp.dot(jnp.dot(jnp.transpose(W), B), W) - jnp.dot(jnp.dot(jnp.transpose(W), A), Wt))


def gamma(W, A, B):
    return jnp.dot(jnp.linalg.inv(jnp.dot(jnp.dot(jnp.transpose(W), B), W)),
                   jnp.dot(jnp.dot(jnp.transpose(W), A), W))


def GenELinK_update(W, A, B, lr, mu, iterations):
    W = agd_solve(obj, A, B, W, x=jnp.dot(W, gamma(W, A, B)), lr=lr, mu=mu, iterations=iterations,
                  in_axes=(None, 0, 0, 0))
    # W = gd_solve(obj, A, B, W, x=jnp.dot(W, gamma(W, A, B)), lr=lr, iterations=iterations)
    return gram_schmidt_matrix(W, B)


def GenELinK(A, B, k, iterations=1000, random_state=0, verbose=False, X=None, Y=None):
    p = X.shape[1]
    key = random.PRNGKey(random_state)
    d = A.shape[1]
    beta = jnp.linalg.norm(B)
    alpha = jnp.min(jnp.abs(jnp.linalg.eig(B)[0]))
    Q = beta / alpha
    mu = (jnp.sqrt(Q) - 1) / (jnp.sqrt(Q) + 1)
    lr = 1 / beta
    W = random.normal(key, (d, k))
    W = gram_schmidt_matrix(W, B)
    for i in range(iterations):
        W = GenELinK_update(W, A, B, lr=lr, mu=mu, iterations=iterations)
        if verbose:
            key = random.PRNGKey(random_state)
            U = random.normal(key, (k, int(k / 2)))
            Wx = gram_schmidt_matrix(jnp.dot(W[:p], U), B[:p, :p])
            Wy = gram_schmidt_matrix(jnp.dot(W[p:], U), B[p:, p:])
            print(f'iteration {i}: {jnp.trace(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy)))}')
    return W


# Run the update step iteratively across all eigenvectors
def calc_ccalin(X, Y, k, iterations=1000, random_state=0, verbose=False):
    n = X.shape[0]
    A = jnp.hstack((X, Y))
    A = jnp.dot(jnp.transpose(A), A) / n
    B = jsp.linalg.block_diag(jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y)) / n
    A = A - B
    W = GenELinK(A, B, 2 * k, iterations=iterations, random_state=random_state, verbose=verbose, X=X, Y=Y)
    key = random.PRNGKey(random_state)
    U = random.normal(key, (2 * k, k))
    Wx = jnp.linalg.qr(jnp.dot(W[:A.shape[1]], U))
    Wy = jnp.linalg.qr(jnp.dot(W[A.shape[1]:], U))
    return calc_eigenvalues(X, Y, Wx, Wy), Wx, Wy
