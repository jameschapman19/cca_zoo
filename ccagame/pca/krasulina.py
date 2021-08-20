"""
Exponentially convergent stochastic k-PCA without variance reduction
https://arxiv.org/pdf/1904.01750.pdf
"""
# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit

from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(2))
def update(u, X, lr=0.1):
    du = (X - X @ u @ u.T).T @ X @ u
    vhat = u + lr * du
    return jnp.linalg.qr(vhat)[0]


def calc_krasulina(X, k, lr=1e-1, iterations=100, initialization='uniform',
                   random_state=0):
    U = initialize(X, k, type=initialization, random_state=random_state)
    obj = []
    for i in range(iterations):
        U = update(U, X, lr=lr)
        obj.append(TV(X, U))
        print(f'iteration {i}: {obj[-1]}')
    return TV(X, U), U, obj
