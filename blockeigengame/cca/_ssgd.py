from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from ._ccamixin import _CCAMixin
from ._utils import _get_AB, _split_eigenvector
from jax import jit
from .._baseexperiment import _BaseExperiment


class SSGD(_BaseExperiment, _CCAMixin):
    def __init__(self, mode, init_rng, config):
        super(SSGD, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._W = jax.random.normal(
            self.init_rng, (config.n_components, self.dims[0] + self.dims[1])
        )
        self._W = self._W / jnp.linalg.norm(self._W, keepdims=True, axis=1)
        self._grads = self._grads
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )
        self._optimizer = optax.sgd(learning_rate=learning_rate)
        self._opt_state = self._optimizer.init(self._W)
        self.learning_rate = learning_rate

    def _update(self, views, global_step):
        X_i, Y_i = views
        grad = self._grads(X_i, Y_i, self._W)
        self._W, self._opt_state = self._update_with_grads(
            self._W, grad, self._opt_state
        )
        self._U, self._V = _split_eigenvector(self._W, self.dims[0])

    @staticmethod
    def _grads(X_i, Y_i, V):
        A, B = _get_AB(X_i, Y_i)
        return (V @ B @ V.T * A @ V.T - V @ A @ V.T * B @ V.T).T

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        updates, opt_state = self._optimizer.update(grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        ui_new /= jnp.linalg.norm(ui_new, keepdims=True)
        return ui_new, opt_state
