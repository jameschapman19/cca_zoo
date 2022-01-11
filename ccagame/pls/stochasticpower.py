from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from jax import jit

from . import PLSExperiment


class StochasticPower(PLSExperiment):
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
        super(StochasticPower, self).__init__(
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
        self._U = jax.random.normal(self.init_rng, (self.n_components, self.dims[0]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.init_rng, (self.n_components, self.dims[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._optimizer = optax.sgd(
            learning_rate=learning_rate
        )
        self._opt_state_x = self._optimizer.init(self._U)
        self._opt_state_y = self._optimizer.init(self._V)

    def _update(self, views, global_step):
        X_i, Y_i = views
        grads_x, grads_y = self._grads(X_i, Y_i, self._U, self._V)
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )
        self._U = self._orth(self._U)
        self._V = self._orth(self._V)

    @staticmethod
    @jit
    def _grads(X_i, Y_i, U, V):
        C = X_i.T @ Y_i / X_i.shape[0]
        grads_x = C @ V.T
        grads_y = C.T @ U.T
        return grads_x.T, grads_y.T

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        return ui_new, opt_state

    @staticmethod
    @jit
    def _orth(U):
        Qu, Ru = jnp.linalg.qr(U.T)
        Su = jnp.sign(jnp.sign(jnp.diag(Ru)) + 0.5)
        return (Qu @ jnp.diag(Su)).T
