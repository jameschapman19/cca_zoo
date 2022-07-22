from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit

from ._ccamixin import _CCAMixin
from ._utils import _get_target
from .._baseexperiment import _BaseExperiment


class Game(_CCAMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(Game, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._weights = jnp.ones((config.n_components, config.n_components))
        self._weights = self._weights.at[jnp.triu_indices(config.n_components, 1)].set(
            0
        )
        # generates weights for each component on each device
        self._grads = jax.jit(jax.vmap(self._grads, in_axes=(0, 1, 1, None, None, 0, None)))
        self._update_with_grads = jax.jit(
            self._update_with_grads,
        )

    def _init_train(self):
        self._init_ground_truth()
        views = next(self._eval_input)
        self._U = jax.random.normal(
            self.init_rng, (self.config.n_components, views[0].shape[1])
        )
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(
            self.init_rng, (self.config.n_components, views[1].shape[1])
        )
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        self._opt_state_x = self._optimizer.init(self._U)
        self._opt_state_y = self._optimizer.init(self._V)

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views
        Zx, Zy = _get_target(X_i, Y_i, self._U, self._V)
        grads_x, rewards, pens = self._grads(self._U, Zx, Zy, Zx, Zy, self._weights, X_i)
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        Zx, Zy = _get_target(X_i, Y_i, self._U, self._V)
        grads_y, rewards, pens = self._grads(self._V, Zy, Zx, Zy, Zx, self._weights, Y_i)
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    def _grads(ui, zx, zy, Zx, Zy, weights, X):
        rewards = (X.T @ zy) * jnp.dot(zx, zx) / X.shape[0]
        penalties = (X.T @ Zx) @ (jnp.dot(zx, Zy) * weights) / X.shape[0]
        return (rewards - penalties) / X.shape[0], rewards, penalties

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        norm = jnp.linalg.norm(ui_new, axis=1, keepdims=True)
        # norm=jnp.where(norm<1,1,norm)
        ui_new /= norm
        return ui_new, opt_state
