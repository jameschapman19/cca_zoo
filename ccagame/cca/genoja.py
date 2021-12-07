from functools import partial
from os import environ
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit
from . import CCAExperiment


class GenOja(CCAExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        learning_rate=1e-6,
        momentum=0.9,
        nesterov=True,
        batch_size=0,
        **kwargs
    ):
        super(GenOja, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            dims=dims,
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
        self._U = jax.random.normal(self.local_rng, (self.n_components, self.dims[0]))
        self._V = jax.random.normal(self.local_rng, (self.n_components, self.dims[1]))
        self._optimizer_ls = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._optimizer_oja = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state_ls = self._optimizer_ls.init(self._U)
        self._opt_state_oja = self._optimizer_oja.init(self._V)


    # @partial(jit, static_argnums=(0))
    def _update(self, views, global_step):
        X_i, Y_i = views
        X_i = jnp.reshape(X_i, (self.num_devices, -1, self.dims[0]))
        Y_i = jnp.reshape(Y_i, (self.num_devices, -1, self.dims[1]))
        
