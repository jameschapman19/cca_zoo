from jax import jit
import jax.numpy as jnp


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
def _gram_schmidt(W, M):
    # TODO
    raise NotImplementedError
