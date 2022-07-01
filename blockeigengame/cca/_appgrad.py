"""
Appgrad
"""
from functools import partial
from re import I

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jax import jit
from scipy.linalg import sqrtm

from .._baseexperiment import _BaseExperiment
from ._ccamixin import _CCAMixin
from ..datasets._utils import data_stream
from absl import flags

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
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )

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

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, wi, grads, opt_state):
        updates, opt_state = self._optimizer.update(grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @staticmethod
    # @jit
    def _normalize(X_i, U, c):
        n = X_i.shape[0]
        M = (U @ X_i.T @ X_i @ U.T) / n + c * jnp.eye(U.shape[0])
        return (U.T @ jnp.linalg.inv(sqrtm(M))).T
