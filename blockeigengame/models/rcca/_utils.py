import jax.numpy as jnp
from jax import jit


@jit
def _get_AB(X_i, Y_i, tau):
    p = X_i.shape[1]
    n = X_i.shape[0]
    C = jnp.hstack((X_i, Y_i)).T @ jnp.hstack((X_i, Y_i)) / n
    A = C.at[:p, :p].set(0)
    A = A.at[p:, p:].set(0)
    B = C.at[:p, p:].set(0)
    B = B.at[p:, :p].set(0)
    B = B.at[:p, :p].set((1 - tau[0]) * B[:p, :p] + tau[0] * jnp.eye(X_i.shape[1]))
    B = B.at[p:, p:].set((1 - tau[1]) * B[p:, p:] + tau[1] * jnp.eye(Y_i.shape[1]))
    return A, B
