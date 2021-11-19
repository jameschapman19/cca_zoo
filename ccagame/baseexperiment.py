from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from jax._src.random import PRNGKey
from jaxline import utils
from jaxline.experiment import AbstractExperiment

class BaseExperiment(AbstractExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
    ):
        super(BaseExperiment, self).__init__(mode=mode, init_rng=init_rng)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        
        self.batch_size=batch_size
        self.n_components=n_components
        self.data = data
        self.local_rng = jax.random.fold_in(PRNGKey(123), jax.host_id())
        self.num_devices = num_devices
        if self.batch_size == 0:
            self.inputs=next(self.data)
        

    def step(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        """Step function for a Jaxline experiment"""
        if self.batch_size == 0:
            self._update(self.inputs, global_step)
        else:
            inputs = next(self.data)
            self._update(inputs, global_step)
        return self._get_scalars()

    def _get_scalars(self, *args, **kwargs):
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
        return self._get_scalars()
