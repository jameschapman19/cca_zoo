"""
MSG
"""
import jax
import jax.numpy as jnp
from jax import jit

from ._ccamixin import _CCAMixin
from .._baseexperiment import _BaseExperiment
from ..._utils import _capping, incrsvd


class Incremental(_CCAMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(Incremental, self).__init__(mode, init_rng, config)
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
        self._S = jnp.zeros(self.config.n_components)
        self.Uxx = jax.random.normal(
            self.init_rng,
            (self.config.kappa * self.config.n_components, views[0].shape[1]),
        )
        self.Uxx /= jnp.linalg.norm(self.Uxx, axis=1, keepdims=True)
        self.Uyy = jax.random.normal(
            self.init_rng,
            (self.config.kappa * self.config.n_components, views[1].shape[1]),
        )
        self.Uyy /= jnp.linalg.norm(self.Uyy, axis=1, keepdims=True)
        self.Sxx = jnp.ones(self.config.kappa * self.config.n_components)
        self.Syy = jnp.ones(self.config.kappa * self.config.n_components)
        if (
                max(views[0].shape[1], views[1].shape[1])
                * min(views[0].shape[1], views[1].shape[1]) ** 2
        ) < ((self.config.n_components + self.config.batch_size) ** 3):
            self._grads = self._mat_grads
        else:
            self._grads = self._incr_grads

    def _update(self, views, global_step):
        X_i, Y_i = views
        self.Uxx, self.Sxx = self.update_cov(
            self.Uxx, self.config.batch_size * (global_step) * self.Sxx, X_i
        )
        self.Uyy, self.Syy = self.update_cov(
            self.Uyy, self.config.batch_size * (global_step) * self.Syy, Y_i
        )
        self.Sxx /= (self.config.batch_size * (global_step + 1))
        self.Syy /= (self.config.batch_size * (global_step + 1))
        Wx = self._get_w(X_i, self.Uxx, self.Sxx)
        Wy = self._get_w(Y_i, self.Uyy, self.Syy
                         )
        self._U, self._V, self._S = self._grads(
            self._U, self._V, self.config.batch_size * (global_step) * self._S, Wx, Wy
        )
        self._S = _capping(self._S / (self.config.batch_size * (global_step + 1)), self.config.n_components)

    @staticmethod
    @jit
    def update_cov(U, S, X):
        x_projected = X @ U.T
        x_leftover = X - x_projected @ U
        U, _, S = incrsvd(x_projected, x_projected, x_leftover, x_leftover, U, U, S)
        return U, S

    @staticmethod
    @jit
    def _incr_grads(U, V, S, X_i, Y_i):
        x_projected = X_i @ U.T
        x_leftover = X_i - x_projected @ U
        y_projected = Y_i @ V.T
        y_leftover = Y_i - y_projected @ V
        return incrsvd(x_projected, y_projected, x_leftover, y_leftover, U, V, S)

    @staticmethod
    @jit
    def _mat_grads(U, V, S, X_i, Y_i):
        n_components = U.shape[0]
        M_projected = X_i.T @ Y_i + U.T @ jnp.diag(S) @ V
        U, S, Vt = jnp.linalg.svd(M_projected)
        return U[:, :n_components].T, Vt[:n_components, :], S[:n_components]

    @staticmethod
    @jit
    def _get_w(X_i, Uxx, Sxx):
        return X_i @ Uxx.T @ jnp.diag(1 / jnp.sqrt(Sxx + 1e-9)) @ Uxx
