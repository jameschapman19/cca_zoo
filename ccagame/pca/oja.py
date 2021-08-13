# Importing necessary libraries

import jax.numpy as jnp
from jax import random

from .utils import calc_eigenvalues, initialize


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
def update(v, X, lr=0.1):
    dv = jnp.dot(jnp.dot(jnp.transpose(X), X), v)
    vhat = v + lr * dv
    return jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_oja(X, n, lr=1e-1, iterations=100, initialization='random',
             random_state=0):
    V=initialize(X,n,type=initialization,random_state=random_state)

    for i in range(iterations):
        V = update(V, X, lr=lr)
        print(f'iteration {i}: {calc_eigenvalues(X, V)}')
    return calc_eigenvalues(X, V), V
