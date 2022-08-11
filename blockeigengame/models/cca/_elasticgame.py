from functools import partial

import jax.numpy as jnp
from jax import jit
import optax
from ._game import Game


class ElasticGame(Game):
    def __init__(self, mode, init_rng, config):
        super(ElasticGame, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self.alpha = 0
        self.lambda_ = 0.1
        self.learning_rate = self.config.learning_rate

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views
        Zx, Zy = self._get_target(X_i, Y_i, self._U, self._V)
        T = Zx + Zy
        grads_x = self.grads(self._U, Zx, T, Zx, T, self._weights, X_i)
        grads_y = self.grads(self._V, Zy, T, Zy, T, self._weights, Y_i)
        updates, self._opt_state_x = self._optimizer_x.update(
            -grads_x, self._opt_state_x
        )
        self._U = optax.apply_updates(self._U, updates)
        self._U = self.prox(self._U)
        updates, self._opt_state_y = self._optimizer_y.update(
            -grads_y, self._opt_state_y
        )
        self._V = optax.apply_updates(self._V, updates)
        self._V = self.prox(self._V)

    @partial(jit, static_argnums=(0))
    def prox(self, ui):
        t = self.lambda_ * self.learning_rate * (1 - self.alpha)
        ui = jnp.where(ui > t, ui - t, ui)
        ui = jnp.where(ui < -t, ui + t, ui)
        ui = jnp.where(jnp.abs(ui) < t, 0, ui)
        return ui

    @partial(jit, static_argnums=(0))
    def _grads(self, ui, zx, t, Zx, T, weights, X):
        rewards = (X.T @ t) * jnp.dot(zx, zx) / X.shape[0]
        penalties = (X.T @ Zx) @ (jnp.dot(zx, T) * weights) / X.shape[0]
        return (rewards - penalties - 2 * self.alpha * self.lambda_ * ui) / X.shape[0]
