import jax
import jax.numpy as jnp

from ._game import Game
from ..._utils import _invsqrtm


class AlphaGame(Game):
    def __init__(self, mode, init_rng, config):
        super(AlphaGame, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        # generates weights for each component on each device
        self.grads = jax.jit(
            jax.vmap(
                jax.grad(self._utils),
                in_axes=(0, 1, 0, None, None, None, None),
            )
        )
        self._utils = jax.jit(
            jax.vmap(
                self._utils,
                in_axes=(0, 1, 0, None, None, None, None),
            )
        )

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views
        if self.config.batch_size == 0 and global_step == 0:
            self.Cxx = _invsqrtm(views[0])
            self.Cyy = _invsqrtm(views[1])
        elif self.config.batch_size > 0:
            self.Cxx = _invsqrtm(views[0])
            self.Cyy = _invsqrtm(views[1])
        Zx, Zy = self._get_target(X_i @ self.Cxx, Y_i @ self.Cyy, self._U, self._V)
        grads_x = self.grads(self._U, Zy, self._weights, X_i, Zx, Zy, self.Cxx)
        self._U, self._opt_state_x = self.update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        Zx, Zy = self._get_target(X_i @ self.Cxx, Y_i @ self.Cyy, self._U, self._V)
        grads_y = self.grads(self._V, Zx, self._weights, Y_i, Zy, Zx, self.Cyy)
        self._V, self._opt_state_y = self.update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    def _utils(ui, zy, weights, X, Zx, Zy, Cxx):
        zx = X @ Cxx @ ui
        rewards = zx @ zy
        covariance = -((zx @ Zy) ** 2) / jnp.diag(Zx.T @ Zy) @ weights
        grads = rewards + covariance / 2
        return grads
