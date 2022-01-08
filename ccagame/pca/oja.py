from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from jax import jit

from . import PCAExperiment


class Oja(PCAExperiment):
    NON_BROADCAST_CHECKPOINT_ATTRS = {"_U": "U", "_V": "V"}

    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        learning_rate=1e-3,
        momentum=0.9,
        nesterov=True,
        batch_size=0,
        **kwargs
    ):
        super(Oja, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=data,
            batch_size=batch_size,
            **kwargs
        )
        """
        Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """
        Initialization function for a Jaxline experiment.
        """
        self._V = jax.random.normal(self.local_rng, (self.n_components, self.dims))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state = self._optimizer.init(self._U)

    def _update(self, inputs, global_step):
        grads = self._grads(inputs, self._V)
        self._V, self._opt_state = self._update_with_grads(
            self._V, grads, self._opt_state
        )

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
        updates, opt_state = self._optimizer.update(grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        vi_new = jnp.linalg.qr(vi_new.T)[0].T
        return vi_new, opt_state
