from ccagame import pca
from ccagame.utils import data_stream
from abc import abstractmethod

import jax.numpy as jnp
from ccagame.baseexperiment import BaseExperiment
from jax import jit
from functools import partial
import numpy as np
from ..datasets.mnist import mnist_iterator


class PCAExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
        TV=False,
        **kwargs,
    ):
        if data == "mnist":
            (
                self.data,
                self.holdout,
                self.correct_eigenvectors,
                self.dims,
            ) = mnist_iterator(
                batch_size=batch_size, n_components=n_components, pca=True
            )
        else:
            raise ValueError("Data {data} not implemented yet")
        super(PCAExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=self.data,
            batch_size=batch_size,
            **kwargs
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self.TV=TV

    @abstractmethod
    def _update(self, X_i, Y_i, global_step):
        raise NotImplementedError

    def _get_scalars(self):
        scalars={}
        if self.TV:
            scalars['TV']=self._TV(self._V, self.holdout)
        scalars['correct_x']=self._correct_eigenvector_streak(
                self._V, self.correct_eigenvectors
            )
        scalars['subspace']=self._normalized_subspace_distance(
                self._V, self.correct_eigenvectors
            )
        return scalars

    @staticmethod
    @jit
    def _TV(U, X_val):
        dof = X_val.shape[0]
        Zx = X_val @ U.T
        return jnp.sum(jnp.diag(Zx.T @ Zx)) / dof

    def save_outputs(self):
        V = jnp.reshape(self._V, (self.n_components, self.dims))
        np.savetxt("V.csv", V, delimiter=",")

    @staticmethod
    @jit
    def _correct_eigenvector_streak(U, U_correct):
        cosine_similarities_x = jnp.diag(U_correct.T @ U.T)
        x_idx = jnp.where(
            jnp.abs(cosine_similarities_x) > jnp.cos(jnp.pi / 8),
            jnp.ones_like(cosine_similarities_x),
            jnp.zeros_like(cosine_similarities_x),
        )
        return jnp.sum(x_idx)

    @staticmethod
    @jit
    def _normalized_subspace_distance(U, U_correct):
        P = U_correct @ U_correct.T
        U_star = U.T @ U
        return 1 - jnp.trace(U_star @ P) / U_correct.shape[1]
