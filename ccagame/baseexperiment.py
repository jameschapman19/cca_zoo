from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from jax._src.random import PRNGKey
from jaxline import utils
from jaxline.experiment import AbstractExperiment

from ccagame.datasets.mnist import mnist_dataset
from ccagame.datasets.ukbiobank import ukbb_dataset
from ccagame.datasets.xrmb import xrmb_dataset
from ccagame.utils import data_stream, data_stream_UKBB
from jax import jit

class BaseExperiment(AbstractExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
        validate=True,
        path=None,
        num_batches=None,
        **kwargs,
    ):
        super(BaseExperiment, self).__init__(mode=mode, init_rng=init_rng)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""

        self.batch_size = batch_size
        self.n_components = n_components
        self.data = data
        self.local_rng = jax.random.fold_in(PRNGKey(123), jax.host_id())
        self.num_devices = num_devices
        self.validate=validate
        self.data_stream=self._init_data(self.data,self.batch_size,path=path,num_batches=num_batches)

    @abstractmethod
    def _init_ground_truth(self,X,Y=None):
        raise NotImplementedError

    def _init_data(self,data,batch_size,path=None, num_batches=None, **kwargs):
        if data == "mnist":
            X,Y,self.X_val,self.Y_val=mnist_dataset()
        elif data == "xrmb":
            X,Y,self.X_val,self.Y_val=xrmb_dataset()
        elif data == "ukbb":
            X,Y,self.X_val,self.Y_val, batch_ids=ukbb_dataset(num_batches, path, batch_size)
        else:
            raise ValueError("Data {data} not implemented yet")
        self.dims=[X.shape[1],Y.shape[1]]
        if self.validate:
            self._init_ground_truth(X,Y=Y)
        if data=='ukbb':
            return data_stream_UKBB(batch_ids, path, batch_size=batch_size)
        else:
            return data_stream(X, Y=Y, batch_size=batch_size)

    @staticmethod
    #@jit
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
        U_star = U @ U.T
        return 1 - jnp.trace(U_star @ P) / U_correct.shape[1]

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
            inputs = next(self.data_stream)
            self._update(inputs, global_step)
        if self.validate:
            return self._get_scalars()
        else:
            return {}

    def _get_scalars(self):
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
