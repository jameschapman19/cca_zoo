import jax.numpy as jnp
import numpy as np


def gram_schmidt_np(V, B, n_components=3):
    n_components = V.shape[0]
    U = V[:n_components].copy()
    for i in range(n_components):
        T = V[i] @ B @ U[:i].T / np.diag(U[:i] @ B @ U[:i].T)
        U[i] = V[i] - T @ U[:i]
        U[i] /= np.sqrt(U[i] @ B @ U[i].T)
    return U


def gram_schmidt(V, B):
    n_components = V.shape[0]
    U = V.copy()
    for i in range(n_components):
        T = V[i] @ B @ U[:i].T / jnp.diag(U[:i] @ B @ U[:i].T)
        U[i] = V[i] - T @ U[:i]
        U[i] /= jnp.sqrt(U[i] @ B @ U[i].T)
    return U


V = np.random.rand(10, 10)


def spd(p=10):
    B = np.random.rand(p, p)
    U, _, Vt = np.linalg.svd(np.dot(B.T, B))
    X = np.dot(np.dot(U, 1.0 + np.diag(np.random.rand(p))), Vt)
    return X


B = spd(10)

U = gram_schmidt_np(V, B)
# U@B@U.T
print()
