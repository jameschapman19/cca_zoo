import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit


@jit
def mat_pow(mat, pow_, epsilon):
    # Computing matrix to the power of pow (pow can be negative as well)
    [D, V] = jnp.linalg.eigh(mat)
    mat_pow = V @ jnp.diag(jnp.power((D + epsilon), pow_)) @ V.T
    return mat_pow


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
    return V  # V @ B @ V.T


@jit
def _TCC(X, Y, U, V):
    k=U.shape[0]
    Zx = X @ U.T
    Zy = Y @ V.T
    return jnp.sum(jnp.abs(jnp.diag(jnp.corrcoef(Zx,Zy,rowvar=False)[k:,:k])))
