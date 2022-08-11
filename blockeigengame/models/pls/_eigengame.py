from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit
from .._baseexperiment import _BaseExperiment
from ..._utils import _split_eigenvector
from ._plsmixin import _PLSMixin
from ._utils import _get_AB


class EigenGame(_PLSMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(EigenGame, self).__init__(mode, init_rng, config)
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
        # generates weights for each component on each device
        self.grads = jax.jit(
            jax.vmap(self._grads, in_axes=(0, None, None, None, None, 0))
        )

    def _init_train(self):
        self._init_ground_truth()
        views = next(self._eval_input)
        self._W = jax.random.normal(
            self.init_rng,
            (self.config.n_components, views[0].shape[1] + views[1].shape[1]),
        )
        self._W /= jnp.linalg.norm(self._W, axis=1, keepdims=True)
        self._optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        self._opt_state = self._optimizer.init(self._W)
        self._Bu = self._W.copy()
        self._optimizer2 = optax.sgd(learning_rate=self.config.beta)
        self._opt_state2 = self._optimizer2.init(self._Bu)

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views
        X_i, Y_i = views
        w_grad, Bu = self.grads(self._W, self._W, X_i, Y_i, self._Bu, self._weights)
        updates, self._opt_state = self._optimizer.update(-w_grad, self._opt_state)
        self._W = optax.apply_updates(self._W, updates)
        self._W = self._normalize(self._W)
        updates, self._opt_state2 = self._optimizer2.update(
            -(Bu - self._Bu), self._opt_state2
        )
        self._Bu = optax.apply_updates(self._Bu, updates)
        # self._Bu=Bu
        self._U, self._V = _split_eigenvector(self._W, X_i.shape[1])

    @staticmethod
    @jit
    def _normalize(U):
        U /= jnp.linalg.norm(U, axis=1, keepdims=True)
        return U

    @staticmethod
    def _grads(ui, U, X, Y, Bu, weights):
        A, B = _get_AB(X, Y)
        denominator = jnp.diag(Bu @ U.T)
        denominator = jnp.where(denominator > 1e-3, denominator, 1e-3)
        Y = U / jnp.expand_dims(jnp.sqrt(denominator), 1)
        By = Bu / jnp.expand_dims(jnp.sqrt(denominator), 1)
        rewards = (ui.T @ B @ ui) * A @ ui - (ui.T @ A @ ui) * B @ ui
        penalties = (
            (((ui.T @ B @ ui) * By.T) - (jnp.outer(ui.T @ B, By @ ui)))
            * (ui.T @ A @ Y.T)
            @ weights
        )
        return (rewards - penalties), (ui.T @ B)
