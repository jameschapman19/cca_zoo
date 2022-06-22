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
from ..datasets._utils import data_stream
from ._ccamixin import _CCAMixin


class MSG(_CCAMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(MSG, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""

    def _init_train(self):
        self._init_ground_truth()
        views = next(self._train_input)
        self._U = jax.random.normal(self.init_rng, (self.config.n_components, views[0].shape[1]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.init_rng, (self.config.n_components, views[1].shape[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self.Cxx = jnp.zeros((views[0].shape[1], views[0].shape[1]))
        self.Cyy = jnp.zeros((views[1].shape[1], views[1].shape[1]))
        self.M = jnp.zeros((views[0].shape[1], views[1].shape[1]))
        self._optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        self._opt_state_M = self._optimizer.init(self.M)

    def _update(self, views, global_step):
        X_i, Y_i = views
        self.Cxx = ((global_step + 1) - 1) / (global_step+1) * self.Cxx + X_i.T @ X_i / (
            global_step+1
        )
        self.Cyy = ((global_step + 1) - 1) / (global_step+1) * self.Cyy + Y_i.T @ Y_i / (
            global_step+1
        )
        delta = self.invsqrtm(self.Cxx) @ X_i.T @ Y_i @ self.invsqrtm(self.Cyy)
        self.M, self._opt_state_M = self._update_with_grads(
            self.M, delta, self._opt_state_M
        )
        U,_,Vt=jnp.linalg.svd(self.M)
        self._U=U[:,:self.config.n_components].T
        self._V=Vt[:self.config.n_components]
        self.M=self._U.T@self._V

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, wi, grads, opt_state):
        updates, opt_state = self._optimizer.update(grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @staticmethod
    # @jit
    def invsqrtm(C):
        return jnp.linalg.inv(sqrtm(C))
