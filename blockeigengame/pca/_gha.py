import jax
import jax.numpy as jnp
import optax
from functools import partial
from jax import jit
from os import environ, stat

from ._pcamixin import _PCAMixin


class GHA(_PCAMixin):
    def __init__(self, mode, init_rng, config):
        super(GHA, self).__init__(mode, init_rng, config)
        """
        Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """
        Initialization function for a Jaxline experiment.
        """
        self._V = jax.random.normal(self.local_rng, (config.n_components, self.dims))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state = self._optimizer.init(self._V)

    def _update(self, inputs, global_step):
        grads = self._grads(inputs, self._V)
        self._V, self._opt_state = self._update_with_grads(
            self._V, grads, self._opt_state
        )

    @staticmethod
    @jit
    def _grads(inputs, V):
        return V @ inputs.T @ inputs - jnp.triu(V @ inputs.T @ inputs @ V.T) @ V

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, vi, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        vi_new /= jnp.linalg.norm(vi_new, axis=1, keepdims=True)
        return vi_new, opt_state
