from os import environ

import jax.numpy as jnp
import jax
from . import PCAExperiment
from functools import partial
from jax import jit

class Oja(PCAExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        batch_size=0,
        **kwargs
    ):
        super(Oja, self).__init__(
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

    def _update(self, inputs, global_step):
        self._V = self._V @ inputs.T @ inputs
        self._V = (jnp.linalg.qr(self._V.T)[0]).T
