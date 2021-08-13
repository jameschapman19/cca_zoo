import jax.numpy as jnp
from jax import random


def initialize(X, n, type='uniform', random_state=0):
    if type == 'uniform':
        V1 = jnp.ones((n, 1))
        V1 = V1 / jnp.linalg.norm(V1)
    elif type == 'random':
        key = random.PRNGKey(random_state)
        V1 = random.normal(key, (X.shape[1], n))
        V1 = V1 / jnp.linalg.norm(V1)
    else:
        print(f'Initialization "{type}" not implemented')
        return
    return V1


# Calculate eigenvalues once the eigenvectors have been computed
def calc_eigenvalues(X, V1):
    M = jnp.dot(jnp.transpose(X), X)
    n = jnp.size(V1, axis=1)
    eigvals = jnp.zeros((1, n))
    for k in range(n):
        eigvals = eigvals.at[:, k].set(jnp.dot(V1[:, k], jnp.dot(M, V1[:, k].reshape(-1, 1))))
    return eigvals
