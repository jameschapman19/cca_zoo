from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit
from ._eigengame import EigenGame
from ..._utils import _split_eigenvector
from ._utils import _get_AB


class DeterministicEigenGame(EigenGame):
    def __init__(self, mode, init_rng, config):
        super(DeterministicEigenGame, self).__init__(mode, init_rng, config)
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
        self.grads = jax.jit(jax.vmap(self._grads, in_axes=(0, None, None, None, 0)))

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views
        X_i, Y_i = views
        w_grad = self.grads(self._W, self._W, X_i, Y_i, self._weights)
        updates, self._opt_state = self._optimizer.update(-w_grad, self._opt_state)
        self._W = optax.apply_updates(self._W, updates)
        self._W = self._normalize(self._W)
        self._U, self._V = _split_eigenvector(self._W, X_i.shape[1])

    @staticmethod
    def _grads(ui, U, X, Y, weights):
        A, B = _get_AB(X, Y)
        denominator = jnp.diag(U @ B @ U.T)
        Y = U / jnp.expand_dims(jnp.sqrt(denominator), 1)
        rewards = (ui.T @ B @ ui) * A @ ui - (ui.T @ A @ ui) * B @ ui
        penalties = (
            (((ui.T @ B @ ui) * (U @ B).T) - (jnp.outer(ui.T @ B, U @ B @ ui)))
            * (ui.T @ A @ Y.T)
            @ weights
        )
        return rewards - penalties
