from functools import partial

from jax import jit, vmap

from .. import cca, pls
from ..cca._sghagame import SGHAGame
from ._utils import _get_AB
import optax
from ..._utils import _split_eigenvector

class RGame(SGHAGame):
    def __init__(self, mode, init_rng, config):
        super(RGame, self).__init__(mode, init_rng, config)

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grad(X_i, Y_i, self._W,self._W, self._weights, self.config.tau)
        updates, self._opt_state = self._optimizer.update(-w_grad, self._opt_state)
        self._W = optax.apply_updates(self._W, updates)
        self._W = self._normalize(self._W)
        self._U, self._V = _split_eigenvector(self._W, X_i.shape[1])

    @staticmethod
    @jit
    @partial(vmap, in_axes=(None, None, 0, None, 0, None))
    def _grad(X_i, Y_i,w, W,weights, tau):
        A, B = _get_AB(X_i, Y_i,tau)
        rewards=A@w - B@w*(w @ A @ w)
        penalties=(B@W.T) @ ((w @ A @ W.T) * weights)
        return rewards-penalties
    
