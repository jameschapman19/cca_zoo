import functools
from os import environ
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from ccagame.baseexperiment import get_config
from ccagame.pls.experiments import PLSExperiment
from ccagame.utils import data_stream
from datasets.mnist import mnist
from jaxline import platform

CORES = 4
environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
FLAGS = flags.FLAGS


class StochasticPower(PLSExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        dims=1,
        data_stream=None,
        whole_batch=False
    ):
        super(MSG, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            k_per_device=k_per_device,
            dims=dims,
            data_stream=data_stream,
            whole_batch=whole_batch
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._U = jax.random.normal(self.local_rng, (k_per_device, dims[0]))
        self._V = jax.random.normal(self.local_rng, (k_per_device, dims[1]))
        self._optimizer_x = optax.sgd(learning_rate=1e-3, momentum=0.9, nesterov=True)
        self._opt_state_x = self._optimizer_x.init(self._U)
        self._optimizer_y = optax.sgd(learning_rate=1e-3, momentum=0.9, nesterov=True)
        self._opt_state_y = self._optimizer_y.init(self._V)

    def _update(self, views, global_step):
        X_i, Y_i = views
        C=X_i.T@Y_i
        grads_x=self._V@C.T
        grads_y=self._U@C
        updates_x, self._opt_state_x = self._optimizer_x.update(grads_x, self._opt_state_x)
        updates_y, self._opt_state_y = self._optimizer_y.update(grads_y, self._opt_state_y)
        self._U = optax.apply_updates(self._U, updates_x)
        self._V = optax.apply_updates(self._V, updates_y)
        self._U = jnp.linalg.qr(self._U.T)[0].T
        self._V = jnp.linalg.qr(self._V.T)[0].T
        return self._U,self._V


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    X, _, X_te, _ = mnist()
    input_data_iterator = data_stream(X[:, :400], Y=X[:, 400:], batch_size=None)
    k_per_device = 3
    FLAGS.config = get_config(input_data_iterator, CORES, k_per_device, [400, 384])
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, StochasticPower))
