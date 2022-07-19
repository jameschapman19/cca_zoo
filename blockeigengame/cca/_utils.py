import jax.numpy as jnp
from jax import jit


@jit
def _gram_schmidt(V, B):
    n_components = V.shape[0]
    for i in range(n_components):
        T = V[i] @ B @ V[:i].T / jnp.diag(V[:i] @ B @ V[:i].T)
        V = V.at[i].set(V[i] - T @ V[:i])
        V = V.at[i].set(V[i] / jnp.sqrt(V[i] @ B @ V[i].T))
    return V


@jit
def _get_target(X, Y, U, V):
    Zx = X @ U.T
    Zy = Y @ V.T
    return Zx, Zy
