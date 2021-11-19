from os import environ

import jax
import jax.numpy as jnp
from . import PLSExperiment


class Oja(PLSExperiment):
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
            dims=dims,
            n_components=n_components,
            data=data,
            batch_size=batch_size,
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._U = jax.random.normal(self.local_rng, (self.n_components, dims[0]))
        self._V = jax.random.normal(self.local_rng, (self.n_components, dims[1]))

    def _update(self, views, global_step):
        X_i, Y_i = views
        C = X_i.T @ Y_i
        self._U = self._V @ C.T
        self._V = self._U @ C
        self._U = jnp.linalg.qr(self._U.T)[0].T
        self._V = jnp.linalg.qr(self._V.T)[0].T
