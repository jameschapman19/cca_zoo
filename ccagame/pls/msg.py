from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from jax import jit

from . import PLSExperiment


class MSG(PLSExperiment):
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
        super(MSG, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=data,
            batch_size=batch_size,
            **kwargs
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._U = (
            jax.random.normal(self.local_rng, (self.n_components, self.dims[0]))
        )
        self._U = (1 / jnp.linalg.norm(self._U, axis=1) * self._U.T).T
        self._V = (
            jax.random.normal(self.local_rng, (self.n_components, self.dims[1]))
        )
        self._V = (1 / jnp.linalg.norm(self._V, axis=1) * self._V.T).T
        self._M = self._U.T @ self._V
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state = self._optimizer.init(self._M)

    def _update(self, views, global_step):
        X_i, Y_i = views
        grads = self._grads(X_i, Y_i)
        self._M, self._U, self._V, self._opt_state = self._update_with_grads(
            self._M, grads, self._opt_state
        )

    @staticmethod
    @jit
    def _grads(X_i, Y_i):
        return X_i.T @ Y_i

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, Mi, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(grads, opt_state)
        Mi_new = optax.apply_updates(Mi, updates)
        U, _, Vt = jnp.linalg.svd(Mi_new)
        U = U[:, : self.n_components].T
        V = Vt[: self.n_components]
        return Mi_new, U, V, opt_state
