"""
MSG
"""
import jax
import jax.numpy as jnp
from jax import jit

from ._ccamixin import _CCAMixin
from .._baseexperiment import _BaseExperiment
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
        self.Uxx = jnp.zeros(
            (
                4 * self.config.n_components,
                views[0].shape[1],
            )
        )
        self.Uyy = jnp.zeros((4 * self.config.n_components, views[1].shape[1]))
        self.Sxx = jnp.zeros(4 * self.config.n_components)
        self.Syy = jnp.zeros(4 * self.config.n_components)
        self._S = jnp.zeros(self.config.n_components)
        if (
            max(views[0].shape[1], views[1].shape[1])
            * min(views[0].shape[1], views[1].shape[1]) ** 2
        ) < ((self.config.n_components + self.config.batch_size) ** 3):
            self._grads = self._mat_grads
        else:
            self._grads = self._incr_grads

    def _update(self, views, global_step):
        lr = 0.03 / max(1, global_step - 100) ** (1 / 3)
        X_i, Y_i = views
        self.Uxx, self.Sxx = self.update_cov(self.Uxx, self.Sxx * global_step, X_i)
        self.Uyy, self.Syy = self.update_cov(self.Uyy, self.Syy * global_step, Y_i)
        self.Sxx /= global_step + 1
        self.Syy /= global_step + 1
        Wx = X_i @ self.Uxx.T @ jnp.diag(1 / jnp.sqrt(self.Sxx + 1e-9)) @ self.Uxx
        Wy = Y_i @ self.Uyy.T @ jnp.diag(1 / jnp.sqrt(self.Syy + 1e-9)) @ self.Uyy
        self._U, self._V, self._S = self._grads(self._U, self._V, self._S, Wx, Wy, lr)
        self._S = self._project_S(self._S)

    @staticmethod
    @jit
    def update_cov(U, S, X):
        x_hat = X @ U.T
        x_orth = X - x_hat @ U
        U, _, S = incrsvd(x_hat, x_hat, x_orth, x_orth, U, U, S)
        return U, S

    @staticmethod
    @jit
    def _incr_grads(U, V, S, X, Y, learning_rate):
        X = jnp.sqrt(learning_rate) * X
        Y = jnp.sqrt(learning_rate) * Y
        x_hat = X @ U.T
        x_orth = X - x_hat @ U
        y_hat = Y @ V.T
        y_orth = Y - y_hat @ V
        U, V, S = incrsvd(x_hat, y_hat, x_orth, y_orth, U, V, S)
        return U, V, S

    @staticmethod
    @jit
    def _mat_grads(U, V, S, X, Y, learning_rate):
        n_components = len(S)
        M_hat = learning_rate * X.T @ Y + U.T @ V
        U, S, Vt = jnp.linalg.svd(M_hat)
        return U[:, :n_components].T, Vt[:n_components, :], S[:n_components]

    @staticmethod
    @jit
    def _project_S(S):
        return jnp.ones_like(S)
