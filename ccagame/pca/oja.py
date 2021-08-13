# Importing necessary libraries

from functools import partial

import jax.numpy as jnp
from jax import jit

from .utils import calc_eigenvalues, initialize


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
@partial(jit, static_argnums=(2))
def update(u, X, lr=0.1):
    dv = jnp.dot(jnp.dot(jnp.transpose(X), X), u)
    vhat = u + lr * dv
    return jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_oja(X, n, lr=1e-1, iterations=100, initialization='uniform',
             random_state=0):
    U = initialize(X, n, type=initialization, random_state=random_state)
    obj = []
    for i in range(iterations):
        U = update(U, X, lr=lr)
        obj.append(calc_eigenvalues(X, U))
        print(f'iteration {i}: {obj[-1]}')
    return calc_eigenvalues(X, U), U, obj
