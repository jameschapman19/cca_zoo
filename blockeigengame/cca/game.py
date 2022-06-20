from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from ._ccamixin import _CCAMixin
from jax import jit

from ._utils import _get_target
from .._baseexperiment import _BaseExperiment

class Game(_BaseExperiment,_CCAMixin):
    def __init__(self, mode, init_rng, config):
        super(Game, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._weights = jnp.ones((self.n_components, self.n_components)) - jnp.eye(
            self.n_components
        )
        self._weights = self._weights.at[jnp.triu_indices(self.n_components, 1)].set(0)
        # generates weights for each component on each device
        self._U = jax.random.normal(self.init_rng, (self.n_components, self.dims[0]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.init_rng, (self.n_components, self.dims[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        if from_utils:
            self._grads = jax.jit(
                jax.vmap(jax.grad(self._utils), in_axes=(0, 1, 0, None, None, 1))
            )
        else:
            self._grads = jax.jit(
                jax.vmap(self._grads, in_axes=(0, 1, 0, None, None, 1))
            )
        self._utils = jax.vmap(self._utils, in_axes=(0, 1, 0, None, None, 1))
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )
        self._optimizer = optax.sgd(learning_rate=learning_rate)
        self._opt_state_x = self._optimizer.init(self._U)
        self._opt_state_y = self._optimizer.init(self._V)
        self.learning_rate = learning_rate
        self.auxiliary_data = self._init_data_stream(self.batch_size, random_state=1)

    def _update(self, views, global_step):
        X_i, Y_i = views
        Zx, Zy = _get_target(X_i, Y_i, self._U, self._V)
        grads_x = self._grads(self._U, Zx, self._weights, X_i, Zx, Zx + Zy)
        grads_y = self._grads(self._V, Zy, self._weights, Y_i, Zy, Zx + Zy)
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    def _utils(ui, zi, weights, X, Zx, Zy):
        zi = X @ ui
        rewards = zi @ Zy
        covariance = -(((zi @ Zx) ** 2) / jnp.diag(Zx.T @ Zx)) @ weights
        variance = -zi @ zi
        grads = rewards + variance
        return grads

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        ui_new /= jnp.linalg.norm(ui_new, keepdims=True)
        return ui_new, opt_state
