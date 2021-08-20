# Importing necessary libraries

from functools import partial

import jax.numpy as jnp
from jax import jit

from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(2))
def update(u, X, lr=0.1):
    dv = X.T@X@u
    vhat = u + lr * dv
    return jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_oja(X, k, lr=1e-1, iterations=100,
             random_state=0):
    U = initialize(X, k, type='random', random_state=random_state)
    obj = []
    for i in range(iterations):
        U = update(U, X, lr=lr)
        obj.append(TV(X, U))
        print(f'iteration {i}: {obj[-1]}')
    return TV(X, U), U, obj
