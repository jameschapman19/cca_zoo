"""
Appgrad
"""
from functools import partial

import jax
import jax.numpy as jnp
import optax
from absl import flags
from jax import jit
from scipy.linalg import sqrtm

from ._ccamixin import _CCAMixin
from .._baseexperiment import _BaseExperiment

flags.DEFINE_list("c", [0, 0], "batch size")


class AppGrad(_CCAMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(AppGrad, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""

    def _init_train(self):
        self._init_ground_truth()
        views = next(self._train_input)
        self._U = jax.random.normal(
            self.init_rng, (self.config.n_components, views[0].shape[1])
        )
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(
            self.init_rng, (self.config.n_components, views[1].shape[1])
        )
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._U_tilde = jnp.zeros_like(self._U)
        self._V_tilde = jnp.zeros_like(self._V)
        self._optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        self._opt_state_x = self._optimizer.init(self._U_tilde)
        self._opt_state_y = self._optimizer.init(self._V_tilde)

    def _update(self, views, global_step):
        X_i, Y_i = views
        x_grads = self._grad(X_i, Y_i, self._V, self._U_tilde, self.config.c[0])
        self._U_tilde, self._opt_state_x = self._update_with_grads(
            self._U, x_grads, self._opt_state_x
        )
        self._U = self._U_tilde / jnp.linalg.norm(self._U_tilde, axis=1, keepdims=True)
        y_grads = self._grad(Y_i, X_i, self._U, self._V_tilde, self.config.c[1])
        self._V_tilde, self._opt_state_y = self._update_with_grads(
            self._V, y_grads, self._opt_state_y
        )
        self._V = self._V_tilde / jnp.linalg.norm(self._V_tilde, axis=1, keepdims=True)

    @staticmethod
    @jit
    def _grad(X_i, Y_i, V, U_tilde, c):
        n = X_i.shape[0]
        grads = (X_i.T @ (X_i @ U_tilde.T) - X_i.T @ Y_i @ V.T) / n + c * U_tilde.T
        return grads.T

    @staticmethod
    # @jit
    def _normalize(X_i, U, c):
        n = X_i.shape[0]
        M = (U @ X_i.T @ X_i @ U.T) / n + c * jnp.eye(U.shape[0])
        return (U.T @ jnp.linalg.inv(sqrtm(M))).T
