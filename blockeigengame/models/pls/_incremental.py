import jax
import jax.numpy as jnp
from jax import jit

from ._plsmixin import _PLSMixin
from .._baseexperiment import _BaseExperiment
from ..._utils import _capping
from ..._utils import incrsvd


class Incremental(_PLSMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(Incremental, self).__init__(mode, init_rng, config)

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
        self._S = jnp.zeros(self.config.n_components)
        if (
            max(views[0].shape[1], views[1].shape[1])
            * min(views[0].shape[1], views[1].shape[1]) ** 2
        ) < ((self.config.n_components + self.config.batch_size) ** 3):
            self._grads = self._mat_grads
        else:
            self._grads = self._incr_grads

    def _update(self, views, global_step):
        X_i, Y_i = views
        self._U, self._V, self._S = self._mat_grads(
            X_i, Y_i, self._U, self._V, self.config.batch_size * (global_step) * self._S
        )
        self._S = _capping(
            self._S / (self.config.batch_size * (global_step + 1)),
            self.config.n_components,
        )

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
