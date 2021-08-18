import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from jax import random

def TCC(X,Y,Wx,Wy)

def gram_schmidt_matrix(W, M):
    for k in range(W.shape[1]):
        C = jnp.zeros((W.shape[0], k))
        for j in range(k):
            C = C.at[:, j].set(jnp.dot(W[:, j], jnp.dot(jnp.transpose(W[:, k]), jnp.dot(M, W[:, j]))))
        W = W.at[:, k].set(W[:, k] - jnp.sum(C, axis=1))
        W = W.at[:, k].set(W[:, k] / jnp.sqrt(jnp.dot(jnp.transpose(W[:, k]), jnp.dot(M, W[:, k]))))
    return W


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


def initialize(X, Y, k, type='uniform', random_state=0):
    if type == 'svd':
        U1, _, V1 = jnp.linalg.svd(X.T @ Y)
        U1 = U1[:, :k]
        V1 = V1[:, :k]
    elif type == 'uniform':
        U1 = jnp.ones((X.shape[1], k))
        V1 = jnp.ones((Y.shape[1], k))
        U1 = U1 / jnp.linalg.norm(U1, axis=0)
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    elif type == 'random':
        key = random.PRNGKey(random_state)
        key, subkey = random.split(key)
        U1 = random.normal(key, (X.shape[1], k))
        V1 = random.normal(subkey, (Y.shape[1], k))
        U1 = U1 / jnp.linalg.norm(U1, axis=0)
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    else:
        print(f'Initialization "{type}" not implemented')
        return
    return U1, V1


def initialize_gep(X, Y):
    n = X.shape[0]
    A = jnp.hstack((X, Y))
    A = jnp.dot(jnp.transpose(A), A) / n
    B = jsp.linalg.block_diag(jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y)) / n
    A = A - B
    return A, B
