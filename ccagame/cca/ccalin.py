"""
Efficient Algorithms for Large-scale Generalized Eigenvector
Computation and Canonical Correlation Analysis
https://export.arxiv.org/pdf/1604.03930
"""
# Importing necessary libraries
import jax.numpy as jnp
from jax import random

from ccagame.solver import agd_solve
from .utils import calc_eigenvalues, gram_schmidt_matrix, initialize_gep, TCC


def obj(W, A, B, Wt):
    return jnp.trace(0.5 * jnp.dot(jnp.dot(jnp.transpose(W), B), W) - jnp.dot(jnp.dot(jnp.transpose(W), A), Wt))


def gamma(W, A):
    return jnp.dot(jnp.dot(jnp.transpose(W), A), W)


def GenELinK_update(W, A, B, lr, mu, iterations):
    W = jnp.squeeze(
        agd_solve(obj, A, B, W, x=jnp.expand_dims(jnp.dot(W, gamma(W, A)), 0), lr=lr, mu=mu, iterations=iterations,
                  in_axes=(0, None, None, None)), 0)
    # W = jnp.squeeze(agd_solve(obj, A, B, W, x=jnp.expand_dims(jnp.dot(W, gamma(W, A)),0), lr=lr, iterations=iterations,
    #                   in_axes=(0, None, None, None)),0)
    return gram_schmidt_matrix(W, B)


def GenELinK(A, B, k, iterations=1000, random_state=0, verbose=False, X=None, Y=None):
    p = X.shape[1]
    d = A.shape[1]
    key = random.PRNGKey(random_state)
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
            print(f'iteration {i}: {TCC(X, Y, Wx, Wy)}')
    return W


# Run the update step iteratively across all eigenvectors
def calc_ccalin(X, Y, k, iterations=1000, random_state=0, verbose=False):
    p = X.shape[1]
    A, B = initialize_gep(X, Y)
    W = GenELinK(A, B, 2 * k, iterations=iterations, random_state=random_state, verbose=verbose, X=X)
    key = random.PRNGKey(random_state)
    U = random.normal(key, (2 * k, k))
    Wx = gram_schmidt_matrix(jnp.dot(W[:p], U), B[:p, :p])
    Wy = gram_schmidt_matrix(jnp.dot(W[p:], U), B[p:, p:])
    return calc_eigenvalues(X, Y, Wx, Wy), Wx, Wy
