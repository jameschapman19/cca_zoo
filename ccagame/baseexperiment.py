from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from absl import flags
from jax._src.random import PRNGKey
from jaxline import utils
from jaxline.base_config import get_base_config
from jaxline.experiment import AbstractExperiment


def get_config(
    data_stream,
    devices,
    dims,
    k_per_device=1,
    log_tensors_interval=1,
    log_train_data_interval=1,
    whole_batch=False
):
    """Return config object for training."""
    config = get_base_config()
    config.experiment_kwargs = {
        "k_per_device": k_per_device,
        "num_devices": devices,
        "dims": dims,
        "data_stream": data_stream,
        "whole_batch": whole_batch
    }
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = True
    config.log_tensors_interval = log_tensors_interval
    config.log_train_data_interval = log_train_data_interval
    config.lock()
    return config


class BaseExperiment(AbstractExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        data_stream=None,
        whole_batch=False
    ):
        super(BaseExperiment, self).__init__(mode=mode, init_rng=init_rng)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._total_k = k_per_device * num_devices
        self.data_stream = data_stream
        self.local_rng = jax.random.fold_in(PRNGKey(123), jax.host_id())
        self._num_devices = num_devices
        self.whole_batch = whole_batch
        if self.whole_batch:
            self.inputs=None

    def step(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        """Step function for a Jaxline experiment"""
        if self.whole_batch:
            outputs = self._update(self.inputs, global_step)
        else:
            inputs = next(self.data_stream)
            outputs = self._update(inputs, global_step)
        return self._get_scalars(outputs)

    def _get_scalars(self, outputs):
        return {}

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
