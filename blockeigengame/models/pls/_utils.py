import jax.numpy as jnp
from jax import jit


@jit
def _get_AB(X_i, Y_i):
    p = X_i.shape[1]
    n = X_i.shape[0]
    C = jnp.hstack((X_i, Y_i)).T @ jnp.hstack((X_i, Y_i)) / n
    A = C.at[:p, :p].set(0)
    A = A.at[p:, p:].set(0)
    B = jnp.eye(A.shape[0])
    return A,B
