import jax.numpy as jnp
from jax import random

from .numpy import calc_numpy


def initialize(X, n, type='uniform', random_state=0):
    if type == 'uniform':
        V1 = jnp.ones((X.shape[1], n))
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    elif type == 'random':
        key = random.PRNGKey(random_state)
        V1 = random.normal(key, (X.shape[1], n))
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    else:
        print(f'Initialization "{type}" not implemented')
        return
    return V1


def TV(X, Wx):
    k = Wx.shape[1]
    return jnp.sum(calc_numpy(jnp.dot(X, Wx), k)[0])
