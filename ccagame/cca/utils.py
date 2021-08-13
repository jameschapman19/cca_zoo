import jax.numpy as jnp
from jax import random


def initialize(X, Y, n, type='uniform', random_state=0):
    if type == 'svd':
        U1, _, V1 = jnp.linalg.svd(X.T @ Y)
        U1 = U1[:, :n]
        V1 = V1[:, :n]
    elif type == 'uniform':
        U1 = jnp.ones((X.shape[1], n))
        V1 = jnp.ones((Y.shape[1], n))
        U1 = U1 / jnp.linalg.norm(U1, axis=0)
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    elif type == 'random':
        key = random.PRNGKey(random_state)
        key, subkey = random.split(key)
        U1 = random.normal(key, (X.shape[1], n))
        V1 = random.normal(subkey, (Y.shape[1], n))
        U1 = U1 / jnp.linalg.norm(U1, axis=0)
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    else:
        print(f'Initialization "{type}" not implemented')
        return
    return U1, V1
