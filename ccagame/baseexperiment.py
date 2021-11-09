import functools
from abc import abstractmethod
from os import environ
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from ccagame.utils import data_stream
from datasets.mnist import mnist
from jax._src.random import PRNGKey
from jaxline import platform, utils
from jaxline.base_config import get_base_config
from jaxline.experiment import AbstractExperiment

# CORES = 2
# environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
FLAGS = flags.FLAGS


def get_config(data_stream, CORES, dims):
    """Return config object for training."""
    config = get_base_config()
    config.experiment_kwargs = {
        "k_per_device": 3,
        "num_devices": CORES,
        "dims": dims,
        "axis_index_groups": None,
        "data_stream": data_stream,
    }
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = True
    config.log_tensors_interval = 1
    config.log_train_data_interval = 1
    config.lock()
    return config


class BaseExperiment(AbstractExperiment):
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
        super(BaseExperiment, self).__init__(mode=mode, init_rng=init_rng)
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

    def step(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        """Step function for a Jaxline experiment"""
        inputs = next(self.data_stream)
        outputs = self._update(inputs, global_step)
        return outputs

    @abstractmethod
    def _update(self, inputs, global_step):
        raise NotImplementedError

    def evaluate(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        return {}
