import functools
from abc import abstractmethod
from os import environ
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from absl import app, flags
from ccagame.utils import data_stream
from datasets.mnist import mnist
from jax._src.random import PRNGKey
from jaxline import platform, utils
from jaxline.base_config import get_base_config
from jaxline.experiment import AbstractExperiment

import numpy as np

CORES = 2
environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
input_data_iterator = None
FLAGS = flags.FLAGS
X, _, X_te, _ = mnist()
input_data_iterator = data_stream(X, Y=None, batch_size=None)


def get_config():
    """Return config object for training."""
    config = get_base_config()
    config.experiment_kwargs = {
        "k_per_device": 3,
        "num_devices": CORES,
        "dims": X.shape[1],
        "axis_index_groups": None,
        "data_stream": input_data_iterator,
    }
    config.checkpoint_dir = "C:/Users/chapm/PycharmProjects/ccagame/jaxlog"
    config.train_checkpoint_all_hosts = True
    config.lock()
    return config


class PLSExperiment(AbstractExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        dims=1,
        axis_index_groups=None,
        data_stream=None,
    ):
        super(PLSExperiment, self).__init__(mode=mode, init_rng=init_rng)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._total_k = k_per_device * num_devices
        self._dims = dims
        self.data_stream = data_stream
        local_rng = jax.random.fold_in(PRNGKey(123), jax.host_id())
        # generates a key for each device
        keys = jax.random.split(local_rng, num_devices)
        self._optimizer = optax.sgd(learning_rate=1e-4, momentum=0.9, nesterov=True)
        self._opt_state = self._optimizer.init(self._V)

    def step(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        """Step function for a Jaxline experiment"""
        X_i, Y_i = next(input_data_iterator)
        self._update(X_i, Y_i, global_step)
        return {}

    @abstractmethod
    def _update(self, X_i, Y_i, global_step):
        raise NotImplementedError

    def evaluate(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        return {}


if __name__ == "__main__":
    FLAGS.config = get_config()
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, PLSExperiment))
