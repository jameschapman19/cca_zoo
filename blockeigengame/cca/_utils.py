from functools import partial
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from scipy.linalg import eigh
import numpy as np


@jit
def mat_pow(mat, pow_, epsilon):
    # Computing matrix to the power of pow (pow can be negative as well)
    U, S, Vt = jnp.linalg.svd(mat)
    return U @ jnp.diag(jnp.power((S + epsilon), pow_)) @ Vt


@jit
def _gram_schmidt(V, B):
    n_components = V.shape[0]
    for i in range(n_components):
        T = V[i] @ B @ V[:i].T / jnp.diag(V[:i] @ B @ V[:i].T)
        V = V.at[i].set(V[i] - T @ V[:i])
        V = V.at[i].set(V[i] / jnp.sqrt(V[i] @ B @ V[i].T))
    return V  # V @ B @ V.T


@jit
def _get_target(X, Y, U, V):
    Zx = X @ U.T
    Zy = Y @ V.T
    return Zx, Zy
