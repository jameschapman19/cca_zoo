import jax.numpy as jnp
from jax import jit


@jit
def _get_AB(X_i, Y_i):
    p = X_i.shape[1]
    n = X_i.shape[0]
    C = jnp.hstack((X_i, Y_i)).T @ jnp.hstack((X_i, Y_i)) / n
    A = C.at[:p, :p].set(0)
    A = A.at[p:, p:].set(0)
    B = C.at[:p, p:].set(0)
    B = B.at[p:, :p].set(0)
    return A, B


@jit
def _gram_schmidt(V, B):
    n_components = V.shape[0]
    for i in range(n_components):
        T = V[i] @ B @ V[:i].T / jnp.diag(V[:i] @ B @ V[:i].T)
        V = V.at[i].set(V[i] - T @ V[:i])
        V = V.at[i].set(V[i] / jnp.sqrt(V[i] @ B @ V[i].T))
    return V
