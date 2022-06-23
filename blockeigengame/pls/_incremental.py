from os import stat
import jax
import jax.numpy as jnp
import optax

from ._utils import incrsvd
from ._plsmixin import _PLSMixin
from .._baseexperiment import _BaseExperiment
from jax import jit


class Incremental(_PLSMixin,_BaseExperiment):
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
        self._U = jax.random.normal(self.init_rng, (self.config.n_components, views[0].shape[1]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.init_rng, (self.config.n_components, views[1].shape[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        self._opt_state_x = self._optimizer.init(self._U)
        self._opt_state_y = self._optimizer.init(self._V)

    def _update(self, views, global_step):
        X_i, Y_i = views
        self._U, self._V, self._S = self._grads(X_i, Y_i, self._U, self._V, self._S)

    @staticmethod
    @jit
    def _incr_grads(X_i, Y_i, U, V, S):
        x_hat = X_i @ U.T
        x_orth = X_i - x_hat @ U
        y_hat = Y_i @ V.T
        y_orth = Y_i - y_hat @ V
        return incrsvd(x_hat, y_hat, x_orth, y_orth, U, V, S)

    @staticmethod
    @jit
    def _mat_grads(X_i, Y_i, U, V, S):
        n_components = U.shape[0]
        M_hat = X_i.T @ Y_i + U.T @ jnp.diag(S) @ V
        U, S, Vt = jnp.linalg.svd(M_hat)
        return U[:, :n_components].T, Vt[:n_components, :], S[:n_components]
