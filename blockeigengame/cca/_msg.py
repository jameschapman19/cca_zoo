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

from .._baseexperiment import _BaseExperiment
from ..datasets._utils import data_stream
from ._ccamixin import _CCAMixin
from .._utils import invsqrtm
from ..pls._utils import incrsvd


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
        self._U = jax.random.normal(
            self.init_rng, (self.config.n_components, views[0].shape[1])
        )
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(
            self.init_rng, (self.config.n_components, views[1].shape[1])
        )
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self.Uxx = jnp.zeros((views[0].shape[1], views[0].shape[1]))
        self.Uyy = jnp.zeros((views[1].shape[1], views[1].shape[1]))
        self.Sxx = jnp.zeros(self.config.n_components)
        self.Syy = jnp.zeros(self.config.n_components)
        self.M = jnp.zeros((views[0].shape[1], views[1].shape[1]))
        self._optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        self._opt_state_M = self._optimizer.init(self.M)

    def _update(self, views, global_step):
        X_i, Y_i = views
        self.Uxx, self.Sxx = self.update_cov(self.Uxx, self.Sxx, X_i)
        self.Uyy, self.Syy = self.update_cov(self.Uxx, self.Sxx, Y_i)
        delta = (
            self.Uxx
            @ jnp.diag(1/jnp.sqrt(self.Sxx))
            @ X_i.T
            @ Y_i
            @ self.Uyy
            @ jnp.diag(1/jnp.sqrt(self.Syy))
        )
        self.M, self._opt_state_M = self._update_with_grads(
            self.M, delta, self._opt_state_M
        )
        U, _, Vt = jnp.linalg.svd(self.M)
        self._U = U[:, : self.config.n_components].T
        self._V = Vt[: self.config.n_components]
        self.M = self._U.T @ self._V

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, wi, grads, opt_state):
        updates, opt_state = self._optimizer.update(grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @staticmethod
    def update_cov(U, S, X):
        n_components=len(S)
        x_hat = X @ U.T
        x_orth = X - x_hat @ U
        U_, _, S = incrsvd(x_hat, x_hat, x_orth, x_orth, S)
        U = U_[:, :n_components].T @ jnp.vstack((U, x_orth / jnp.linalg.norm(x_orth)))
        S = S[:n_components]
        return U,S
