from os import environ

import jax.numpy as jnp
import jax
from . import PCAExperiment
import optax


class GHA(PCAExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        batch_size=0,
        learning_rate=1e-3,
        momentum=0.9,
        nesterov=True,
        **kwargs
    ):
        super(GHA, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            dims=dims,
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
        self._V = jax.random.normal(self.local_rng, (self.n_components, dims))
        self._optimizer =  optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        self._opt_state = self._optimizer.init(self._V)

    def _update(self, inputs, global_step):
        dv = (
            self._V @ inputs.T @ inputs
            - jnp.triu(self._V @ inputs.T @ inputs @ self._V.T) @ self._V
        )
        self._V, self._opt_state = self._update_with_grads(
            self._V, dv, self._optimizer, self._opt_state
        )
        return self._V

    def _update_with_grads(self, vi, grads, opt, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = opt.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        vi_new = jnp.diag(1 / jnp.linalg.norm(vi_new, axis=1)) @ vi_new
        return vi_new, opt_state