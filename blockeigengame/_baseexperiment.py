from abc import abstractmethod
from typing import Generator

import jax
import jax.numpy as jnp
from jaxline.experiment import AbstractExperiment

from .datasets import (
    exponential_dataset,
    linear_dataset,
    mnist_dataset,
    ukbb_dataset,
    xrmb_dataset,
    mediamill_dataset,
)
from .datasets._utils import data_stream


class _BaseExperiment(AbstractExperiment):
    def __init__(self, mode, init_rng, config):
        super(_BaseExperiment, self).__init__(mode, init_rng)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""

        self.mode = mode
        self.init_rng = init_rng
        self.config = config

        # Input pipelines.
        self._train_input = None
        self._eval_input = None

    def _init_train(self):
        pass

    def _build_input(self) -> Generator:
        """See base class."""
        num_devices = jax.device_count()
        global_batch_size = self.config.batch_size
        per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

        if ragged:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"num devices {num_devices}"
            )
        self.X, self.Y, self.X_val, self.Y_val = self._load_data()
        self._train_input = data_stream(
            self.X,
            self.Y,
            batch_size=per_device_batch_size,
            random_state=self.config.random_state,
        )
        self._eval_input = data_stream(
            self.X_val,
            self.Y_val,
            batch_size=self.config.batch_size,
            random_state=self.config.random_state,
        )

    def _load_data(
            self,
    ):
        if self.config.data == "mnist":
            X, Y, X_val, Y_val = mnist_dataset(
                model=self.config.model, random_state=self.config.random_state
            )
        elif self.config.data == "xrmb":
            X, Y, X_val, Y_val = xrmb_dataset()
        elif self.config.data == "mediamill":
            X, Y, X_val, Y_val = mediamill_dataset()
        elif self.config.data == "linear":
            X, Y, X_val, Y_val = linear_dataset(
                self.config.n_components,
                model=self.config.model,
                random_state=self.config.random_state,
            )
        elif self.config.data == "exponential":
            X, Y, X_val, Y_val = exponential_dataset(
                self.config.n_components,
                model=self.config.model,
                random_state=self.config.random_state,
            )
        elif self.config.data == "ukbb":
            X, Y, X_val, Y_val = ukbb_dataset(self.config.path)
        else:
            raise ValueError("Data {data} not implemented yet")
        return X, Y, X_val, Y_val

    def step(
            self, global_step: jnp.ndarray, rng: jnp.ndarray, *unused_args, **unused_kwargs
    ):
        if self._train_input is None:
            self._build_input()
            self._init_train()

        inputs = next(self._train_input)
        self._update(inputs, global_step)
        if global_step == 0 or (global_step + 1) % self.config.val_interval == 0:
            return self._get_scalars(global_step)
        else:
            return {}

    def _get_scalars(self, global_step):
        return {}

    @abstractmethod
    def _update(self, inputs, global_step):
        raise NotImplementedError

    def evaluate(self, global_step: jnp.ndarray, rng: jnp.ndarray, **unused_kwargs):
        return self._get_scalars()
