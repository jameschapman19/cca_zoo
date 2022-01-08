import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from scipy.linalg import eigh
import numpy as np


@jit
def mat_pow(mat, pow_, epsilon):
    # Computing matrix to the power of pow (pow can be negative as well)
    U, S, Vt = jnp.linalg.svd(mat)
    mat_pow = U @ jnp.diag(jnp.power((S + epsilon), pow_)) @ Vt
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


# @jit
def _TCC(X, Y, U, V):
    k = U.shape[0]
    n = X.shape[0]
    Zx = X @ U.T
    Zy = Y @ V.T
    all = jnp.hstack((Zx, Zy))
    C = all.T @ all
    D = jsp.linalg.block_diag(Zx.T @ Zx, Zy.T @ Zy)
    C = C - D
    D = D + 1e-3 * jnp.eye(C.shape[0])
    p = C.shape[0]
    try:
        return eigh(C, D, subset_by_index=[p - k, p - 1])[0].sum()
    except:
        return np.nan
