import jax
import jax.numpy as jnp
import optax
from functools import partial
from jax import jit
from os import environ

from ._pcamixin import _PCAMixin


class Oja(_PCAMixin):
    def __init__(self, mode, init_rng, config):
        super(Oja, self).__init__(mode, init_rng, config)
        """
        Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """
        Initialization function for a Jaxline experiment.
        """
        self._V = jax.random.normal(self.init_rng, (config.n_components, self.dims))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._optimizer = optax.sgd(learning_rate=learning_rate)
        self._opt_state = self._optimizer.init(self._V)

    def _update(self, inputs, global_step):
        grads = self._grads(inputs, self._V)
        self._V, self._opt_state = self._update_with_grads(
            self._V, grads, self._opt_state
        )
        self._V = self._orth(self._V)

    @staticmethod
    @jit
    def _grads(X_i, V):
        C = X_i.T @ X_i / X_i.shape[0]
        grads = V @ C
        return grads

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, vi, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        return vi_new, opt_state

    @staticmethod
    @jit
    def _orth(U):
        Qu, Ru = jnp.linalg.qr(U.T)
        Su = jnp.sign(jnp.sign(jnp.diag(Ru)) + 0.5)
        return (Qu @ jnp.diag(Su)).T
