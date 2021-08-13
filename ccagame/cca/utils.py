import jax.numpy as jnp
from jax import random
from jax import jit

# Calculate eigenvalues once the eigenvectors have been computed
@jit
def calc_eigenvalues(X, Y, U1, V1):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    C_xx = jnp.dot(jnp.transpose(X), X)
    C_yy = jnp.dot(jnp.transpose(Y), Y)
    n = jnp.size(V1, axis=1)
    eigvals = jnp.zeros((1, n))
    for k in range(n):
        eigvals = eigvals.at[:, k].set(jnp.dot(U1[:, k], jnp.dot(C_xy, V1[:, k].reshape(-1, 1))) / (
                jnp.sqrt(jnp.dot(U1[:, k], jnp.dot(C_xx, U1[:, k].reshape(-1, 1)))) * jnp.sqrt(
            jnp.dot(V1[:, k], jnp.dot(C_yy,
                                      V1[:, k].reshape(
                                          -1, 1))))))
    return eigvals


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
