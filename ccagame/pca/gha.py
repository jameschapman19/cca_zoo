# Importing necessary libraries

from functools import partial

import jax.numpy as jnp
from jax import jit

from .utils import initialize, TV


# Update rule to be used for calculating eigenvectors
@partial(jit, static_argnums=(2))
def update(u, X, lr=0.1):
    dv = jnp.dot(jnp.dot(jnp.transpose(X), X), u) - jnp.dot(u, jnp.triu(
        jnp.dot(jnp.transpose(jnp.dot(X, u)), jnp.dot(X, u))))
    vhat = u + lr * dv
    return vhat / jnp.linalg.norm(vhat, axis=0)

def calc_gha(X, k, lr=1e-1, iterations=100, initialization='uniform',
             random_state=0):
    U = initialize(X, k, type=initialization, random_state=random_state)
    obj = []
    for i in range(iterations):
        U = update(U, X, lr=lr)
        obj.append(TV(X, U))
        print(f'iteration {i}: {obj[-1]}')
    return TV(X, U), U, obj
