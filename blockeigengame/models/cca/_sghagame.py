from functools import partial

from jax import jit, vmap

from ._sgha import SGHA
from ._utils import _get_AB
from ..._utils import _split_eigenvector


class SGHAGame(SGHA):
    def __init__(self, mode, init_rng, config):
        super(SGHAGame, self).__init__(mode, init_rng, config)

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grad(X_i, Y_i, self._W, self._W, self._weights)
        self._W, self._opt_state = self._update_with_grads(
            self._W, w_grad, self._opt_state
        )
        self._W = self._normalize(self._W)
        self._U, self._V = _split_eigenvector(self._W, X_i.shape[1])

    @staticmethod
    @jit
    @partial(vmap, in_axes=(None, None, 0, None, 0))
    def _grad(X_i, Y_i, w, W, weights):
        A, B = _get_AB(X_i, Y_i)
        Y = W @ A @ W.T
        return B @ W.T @ Y @ weights - A @ w
