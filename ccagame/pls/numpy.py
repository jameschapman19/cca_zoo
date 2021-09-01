# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from . import _PLS


# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
@partial(jit, static_argnums=(2))
def calc_numpy(X, Y, k):
    C = X.T @ Y
    U, S, Vt = jnp.linalg.svd(C)
    U = lax.dynamic_slice(U, (0, 0), (U.shape[0], k))
    Vt = lax.dynamic_slice(Vt, (0, 0), (Vt.shape[0], k))
    S = lax.dynamic_slice(S.reshape(1, C.shape[0]), (0, 0), (1, k))
    return S, U, Vt.T


class Numpy(_PLS):
    def __init__(self, n_components=2, *, scale=True, copy=True, lr: float = 1, epochs: int = 100,
                 random_state: int = 0, verbose=False):
        super().__init__(n_components, scale=scale, copy=copy)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, X, Y):
        C = X.T @ Y
        U, S, Vt = jnp.linalg.svd(C)
        U = lax.dynamic_slice(U, (0, 0), (U.shape[0], self.n_components))
        Vt = lax.dynamic_slice(Vt, (0, 0), (Vt.shape[0], self.n_components))
        S = lax.dynamic_slice(S.reshape(1, C.shape[0]), (0, 0), (1, self.n_components))
        return S, U, Vt.T
