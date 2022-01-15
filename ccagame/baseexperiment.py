from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit
from jax._src.random import PRNGKey
from jaxline import utils
from jaxline.experiment import AbstractExperiment

from ccagame.datasets import (
    exponential_dataset,
    linear_dataset,
    mnist_dataset,
    ukbb_dataset,
    xrmb_dataset,
)
from ccagame.utils import data_stream, data_stream_UKBB


class BaseExperiment(AbstractExperiment):
    NON_BROADCAST_CHECKPOINT_ATTRS = {"_U": "U", "_V": "V"}

    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
        path=None,
        num_batches=None,
        pca=False,
        cca=False,
        val_interval=0,
        random_state=0,
        **kwargs,
    ):
        super(BaseExperiment, self).__init__(mode=mode, init_rng=init_rng)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""

        self.init_rng=init_rng
        self.batch_size = batch_size
        self.n_components = n_components
        self.data = data
        self.num_devices = num_devices
        self.X, self.Y, self.X_val, self.Y_val, batch_ids = self._init_data(
            self.data,
            self.batch_size,
            path=path,
            num_batches=num_batches,
            pca=pca,
            cca=cca,
            random_state=random_state)
        self.batch_ids = batch_ids
        self.path = path
        self.data_stream = self._init_data_stream(self.batch_size,random_state=random_state)
        self.val_interval = val_interval

    @abstractmethod
    def _init_ground_truth(self, X, Y=None):
        raise NotImplementedError

    def _init_data(
        self,
        data,
        batch_size,
        path=None,
        num_batches=None,
        pca=False,
        cca=False,
        random_state=0,
        **kwargs,
    ):
        batch_ids = None
        if data == "mnist":
            X, Y, X_val, Y_val = mnist_dataset(pca=pca, random_state=random_state)
        elif data == "xrmb":
            X, Y, X_val, Y_val = xrmb_dataset()
        elif data == "linear":
            X, Y, X_val, Y_val = linear_dataset(cca=cca, random_state=random_state)
        elif data == "exponential":
            X, Y, X_val, Y_val = exponential_dataset(cca=cca, random_state=random_state)
        elif data == "ukbb":
            X, Y, X_val, Y_val, batch_ids = ukbb_dataset(num_batches, path, batch_size)
        else:
            raise ValueError("Data {data} not implemented yet")
        return X, Y, X_val, Y_val, batch_ids

    def _init_data_stream(self, batch_size, random_state=0):
        if self.data == "ukbb":
            return data_stream_UKBB(self.batch_ids, self.path, batch_size=batch_size)
        else:
            return data_stream(
                self.X, Y=self.Y, batch_size=batch_size, random_state=random_state
            )

    @staticmethod
    @jit
    def _correct_eigenvector_streak(U, U_correct):
        n_components = U.shape[0]
        cosine_similarities = jnp.diag(
            jnp.corrcoef(U.T, U_correct, rowvar=False)[n_components:, :n_components]
        )
        x_idx = jnp.where(
            jnp.abs(cosine_similarities) > jnp.cos(jnp.pi / 8),
            jnp.ones_like(cosine_similarities),
            jnp.zeros_like(cosine_similarities),
        )
        return jnp.sum(x_idx)

    @staticmethod
    @jit
    def _normalized_subspace_distance(U, U_correct):
        U = U.T / jnp.linalg.norm(U, axis=1)
        P = U_correct @ U_correct.T
        U_star = U @ U.T#P[13,13]
        return 1 - jnp.trace(U_star @ P) / U_correct.shape[1]

    def step(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        """Step function for a Jaxline experiment"""
        inputs = next(self.data_stream)
        self._update(inputs, global_step)
        return self._get_scalars(global_step)

    def _get_scalars(self, global_step):
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
