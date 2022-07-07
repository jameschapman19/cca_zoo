from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit

from ._pcamixin import _PCAMixin


class Game(_PCAMixin):
    def __init__(self, mode, init_rng, config):
        super(Game, self).__init__(mode, init_rng, config)
        """
        Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """
        Initialization function for a Jaxline experiment.
        """
        self._weights = jnp.ones((config.n_components, config.n_components)) - jnp.eye(
            config.n_components
        )
        self._weights = self._weights.at[jnp.triu_indices(config.n_components, 1)].set(
            0
        )
        self._V = jax.random.normal(self.init_rng, (config.n_components, self.dims))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)

        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        self._grads = jax.jit(
            jax.vmap(
                self._grads,
                in_axes=(1, 0, None, None, None),
            )
        )
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state = self._optimizer.init(self._V)

    def _update(self, inputs, global_step):
        Zx = self._get_target(inputs, self._V)
        grads = self._grads(Zx, self._weights, self._V, inputs, Zx)
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads, self._opt_state
        )

    @staticmethod
    @jit
    def _get_target(X, V):
        Z = X @ V.T
        return Z

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, vi, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        vi_new /= jnp.linalg.norm(vi_new, keepdims=True)
        return vi_new, opt_state

    @staticmethod
    def _grads(zi, weights, V, X, Z):
        penalty_grads = -((zi @ Z) * V.T) @ weights
        grads = X.T @ zi + penalty_grads
        return grads / X.shape[0]
