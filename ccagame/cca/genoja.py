"""
Gen-Oja: A Simple and Efficient Algorithm for
Streaming Generalized Eigenvector Computation
https://proceedings.neurips.cc/paper/2018/file/1b318124e37af6d74a03501474f44ea1-Paper.pdf
"""
# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit

from ccagame.cca.utils import initialize, initialize_gep, gram_schmidt_matrix


# Update rule to be used for calculating eigenvectors
#@partial(jit, static_argnums=5)
def update(A, B, W, V, beta, alpha):
    W = W - alpha * (jnp.dot(B, W) - jnp.dot(A, V))
    V = V + beta * W
    return W, V/jnp.linalg.norm(V,axis=0)


# Run the update step iteratively across all eigenvectors
def calc_genoja(X, Y, k, iterations=100,
                alpha=10, beta_0=10, random_state=0):
    p = X.shape[1]
    A, B = initialize_gep(X, Y)
    W, V = initialize(X, Y, k, type='random', random_state=random_state)
    W = jnp.vstack((W, V))
    V = W
    for i in range(iterations):
        W, V = update(A, B, W, V, beta_0 / (1 + 1e-4 * i), alpha)
        Wx = gram_schmidt_matrix(W[:p], B[:p, :p])
        Wy = gram_schmidt_matrix(W[p:], B[p:, p:])
        print(f'iteration {i}: {jnp.sum(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy)))}')
    return jnp.sum(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy))), Wx, Wy
