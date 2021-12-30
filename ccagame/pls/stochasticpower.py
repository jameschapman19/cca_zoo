from os import environ

import jax
import jax.numpy as jnp
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
        self._U = jax.random.normal(self.local_rng, (self.n_components, self.dims[0]))
        self._V = jax.random.normal(self.local_rng, (self.n_components, self.dims[1]))

    def _update(self, views, global_step):
        X_i, Y_i = views
        self._U, self._V = self._grads(X_i, Y_i, self._U, self._V)

    @staticmethod
    @jit
    def _grads(X, Y, U, V):
        C = X.T @ Y
        U = V @ C.T
        V = U @ C
        U = jnp.linalg.qr(U.T)[0].T
        V = jnp.linalg.qr(V.T)[0].T
        return U, V
