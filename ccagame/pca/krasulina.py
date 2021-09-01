"""
Exponentially convergent stochastic k-PCA without variance reduction
https://arxiv.org/pdf/1904.01750.pdf
"""
import time
# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit

from .utils import TV, initialize
# Update rule to be used for calculating eigenvectors
from ..utils import data_stream, get_num_batches


@partial(jit, static_argnums=(2))
def update(u, X, lr=0.1):
    du = (X - X @ u @ u.T).T @ X @ u
    vhat = u + lr * du
    return jnp.linalg.qr(vhat)[0]


def calc_krasulina(X, k, lr=1e-1, epochs=100, initialization='uniform',
                   random_state=0, batch_size=None):
    U = initialize(X, k, type=initialization, random_state=random_state)
    batches = data_stream(X, batch_size=batch_size)
    num_batches = get_num_batches(X, batch_size=batch_size)
    obj = []
    for epoch in range(epochs):
        start_time = time.time()
        for _ in range(num_batches):
            U = update(U, next(batches), lr=lr)
        epoch_time = time.time() - start_time
        obj.append(TV(X, U))
        print(f"Epoch {epoch} in {epoch_time} sec")
        print(f'epoch {epoch}: {obj[-1]}')
    return TV(X, U), U, obj
