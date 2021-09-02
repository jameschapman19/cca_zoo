# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from ccagame.pls._pls import _PLS

#object form
class Numpy(_PLS):
    def __init__(self, n_components=2, *, scale=True, copy=True):
        super().__init__(n_components, scale=scale, copy=copy)

    def _fit(self, X, Y):
        C = X.T @ Y
        U, S, Vt = jnp.linalg.svd(C)
        U = lax.dynamic_slice(U, (0, 0), (U.shape[0], self.n_components))
        Vt = lax.dynamic_slice(Vt, (0, 0), (self.n_components, Vt.shape[0]))
        return U, Vt.T


# function form
@partial(jit, static_argnums=(2))
def calc_numpy(X, Y, k):
    C = X.T @ Y
    U, S, Vt = jnp.linalg.svd(C)
    U = lax.dynamic_slice(U, (0, 0), (U.shape[0], k))
    Vt = lax.dynamic_slice(Vt, (0, 0), (k, Vt.shape[0]))
    S = lax.dynamic_slice(S.reshape(1, C.shape[0]), (0, 0), (1, k))
    return S, U, Vt.T