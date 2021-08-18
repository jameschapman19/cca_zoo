"""
On Constrained Nonconvex Stochastic Optimization: A Case
Study for Generalized Eigenvalue Decomposition
https://proceedings.mlr.press/v89/chen19a/chen19a.pdf
"""
# Importing necessary libraries

import jax.numpy as jnp

from ccagame.cca.utils import initialize, initialize_gep, gram_schmidt_matrix, TCC


# Update rule to be used for calculating eigenvectors
# @partial(jit, static_argnums=5)
def update(A, B, W, lr):
    Y = jnp.dot(jnp.transpose(W), jnp.dot(A, W))
    W = W - lr * (jnp.dot(jnp.dot(B, W), Y) - jnp.dot(A, W))
    return W


# Run the update step iteratively across all eigenvectors
def calc_lagrangeminmax(X, Y, k, iterations=100,
                        lr=100, random_state=0):
    p = X.shape[1]
    A, B = initialize_gep(X, Y)
    W, V = initialize(X, Y, k, type='random', random_state=random_state)
    W = jnp.vstack((W, V))
    for i in range(iterations):
        W = update(A, B, W, lr)
        print(f'iteration {i}: {TCC(X,Y,W[:p],W[p:])}')
        #print(f'iteration {i}: {jnp.sum(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy)))}')
    Wx = gram_schmidt_matrix(W[:p], B[:p, :p])
    Wy = gram_schmidt_matrix(W[p:], B[p:, p:])
    return jnp.sum(jnp.dot(jnp.transpose(Wx), jnp.dot(A[:p, p:], Wy))), Wx, Wy
