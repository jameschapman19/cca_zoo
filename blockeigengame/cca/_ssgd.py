from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from jax import jit

from .._baseexperiment import _BaseExperiment
from ._ccamixin import _CCAMixin
from .._utils import _split_eigenvector, _get_AB


class SSGD(_CCAMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(SSGD, self).__init__(mode, init_rng, config)
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
        self._grads = self._grads
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )

    def _init_train(self):
        self._init_ground_truth()
        views = next(self._train_input)
        self.W = jax.random.normal(
            self.init_rng,
            (self.config.n_components, views[0].shape[1] + views[1].shape[1]),
        )
        self.W = self.W / jnp.linalg.norm(self.W, keepdims=True, axis=1)
        self._optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        self._opt_state = self._optimizer.init(self.W)

    def _update(self, views, global_step):
        X_i, Y_i = views
        grad = self._grads(X_i, Y_i, self.W, self._weights)
        self.W, self._opt_state = self._update_with_grads(self.W, grad, self._opt_state)
        self._U, self._V = _split_eigenvector(self.W, X_i.shape[1])

    @staticmethod
    @jit
    def _grads(X_i, Y_i, V, weights):
        A, B = _get_AB(X_i, Y_i)
        return (A @ V.T @ V @ B @ V.T - (B @ V.T @ V @ A @ V.T)).T

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        ui_new /= jnp.linalg.norm(ui_new, keepdims=True)
        return ui_new, opt_state
