from functools import partial

from jax import jit, vmap

from .. import cca, pls
from ..cca._game import Game


class RGame(Game):
    def __init__(self, mode, init_rng, config):
        super(RGame, self).__init__(mode, init_rng, config)
        self._grads = jit(vmap(self._grads, in_axes=(0, 1, 1, None, None, 0, None, None, None)))

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views  # ((X_i.T@Zx[:,0])*(jnp.dot(Zx[:,0],Zy[:,0])+jnp.dot(Zy[:,0],Zx[:,0]))-(X_i.T@Zy[:,0])*(jnp.dot(Zx[:,0],Zx[:,0])+jnp.dot(Zy[:,0],Zy[:,0])))/X_i.shape[0]
        Zx, Zy = self._get_target(X_i, Y_i, self._U, self._V)
        grads_x = self._grads(
            self._U, Zx, Zy, Zx, Zy, self._weights, X_i, self._U, self.config.tau[0]
        )
        grads_y = self._grads(
            self._V, Zy, Zx, Zy, Zx, self._weights, Y_i, self._V, self.config.tau[1]
        )
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @partial(jit, static_argnums=(0))
    def _grads(self, ui, zx, zy, Zx, Zy, weights, X, U, tau):
        return tau * pls.Game._grads(self, ui, zx, zy, Zx, Zy, weights, X, U) + (
                1 - tau
        ) * cca.Game._grads(self, ui, zx, zy, Zx, Zy, weights, X)
