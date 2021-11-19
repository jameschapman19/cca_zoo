from os import environ
import jax
import jax.numpy as jnp
import optax
from . import PLSExperiment


class StochasticPower(PLSExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
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
        self._U = jax.random.normal(self.local_rng, (self.n_components, dims[0]))
        self._V = jax.random.normal(self.local_rng, (self.n_components, dims[1]))
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state_x = self._optimizer.init(self._U)
        self._opt_state_y = self._optimizer.init(self._V)

    def _update(self, views, global_step):
        X_i, Y_i = views
        C = X_i.T @ Y_i
        grads_x = self._V @ C.T
        grads_y = self._U @ C
        updates_x, self._opt_state_x = self._optimizer.update(
            grads_x, self._opt_state_x
        )
        updates_y, self._opt_state_y = self._optimizer.update(
            grads_y, self._opt_state_y
        )
        self._U = optax.apply_updates(self._U, updates_x)
        self._V = optax.apply_updates(self._V, updates_y)
        self._U = jnp.linalg.qr(self._U.T)[0].T
        self._V = jnp.linalg.qr(self._V.T)[0].T
        return self._U, self._V
