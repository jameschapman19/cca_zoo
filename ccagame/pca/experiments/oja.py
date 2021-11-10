import functools
from os import environ

import jax.numpy as jnp
from absl import app, flags
from ccagame.baseexperiment import get_config
from ccagame.utils import data_stream
from datasets.mnist import mnist
from jaxline import platform
import jax
from ccagame.pca.experiments import PCAExperiment

CORES = 1
environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
FLAGS = flags.FLAGS


class Oja(PCAExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        dims=1,
        data_stream=None,
    ):
        super(Oja, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            k_per_device=k_per_device,
            dims=dims,
            data_stream=data_stream,
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
        self._V = jax.random.normal(self.local_rng, (k_per_device, dims))

    def _update(self, inputs, global_step):
        self._V = self._V @ inputs.T @ inputs
        self._V = (jnp.linalg.qr(self._V.T)[0]).T
        return self._V


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    X, _, X_te, _ = mnist()
    input_data_iterator = data_stream(X, Y=None, batch_size=None)
    FLAGS.config = get_config(input_data_iterator, CORES, X.shape[1])
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, Oja))
