from abc import abstractmethod
from functools import partial
from typing import Optional

import jax.numpy as jnp
from ccagame.baseexperiment import BaseExperiment
from ..datasets.ukbiobank import ukbb_iterator
from ..datasets.xrmb import xrmb_iterator
from jax import jit
import numpy as np
from ..datasets.mnist import mnist_iterator


class CCAExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
        path=None,
        TCC=False,
        **kwargs,
    ):
        if data == "mnist":
            (
                self.data,
                self.holdout,
                self.correct_eigenvectors,
                self.dims,
            ) = mnist_iterator(
                batch_size=batch_size, n_components=n_components, cca=True, p=400
            )
        elif data == "xrmb":
            (
                self.data,
                self.holdout,
                self.correct_eigenvectors,
                self.dims,
            ) = xrmb_iterator(
                batch_size=batch_size, n_components=n_components, cca=True
            )
        elif data == "ukbb":
            self.data, self.holdout, self.dims = ukbb_iterator(
                path, batch_size=batch_size
            )
            self.correct_eigenvectors = None
        else:
            raise ValueError("Data {data} not implemented yet")
        super(CCAExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=self.data,
            batch_size=batch_size,
            **kwargs,
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self.TCC=TCC

    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    def _get_scalars(self):
        scalars={}
        if self.TCC:
            scalars['TCC']=self._TC(self._U, self._V,self.holdout[0],self.holdout[1])
        scalars["correct x"]= self._correct_eigenvector_streak(
            self._U, self.correct_eigenvectors[0]
        )
        scalars["correct y"]= self._correct_eigenvector_streak(
            self._V, self.correct_eigenvectors[1]
        )
        scalars["subspace x"]= self._normalized_subspace_distance(
            self._U, self.correct_eigenvectors[0]
        )
        scalars["subspace y"]= self._normalized_subspace_distance(
            self._V, self.correct_eigenvectors[1]
        )
        return scalars

    @staticmethod
    @jit
    def _TC(U, V,X_val, Y_val):
        Zx = X_val @ U.T
        Zy = (
            Y_val @ V.T
        )
        return jnp.trace(
            jnp.abs(
                jnp.corrcoef(Zx, Zy,rowvar=False)[U.shape[0] :, : U.shape[0]]
            )
        )

    def save_outputs(self):
        np.savetxt("U.csv", self._U, delimiter=",")
        np.savetxt("V.csv", self._V, delimiter=",")

    @staticmethod
    #@jit
    def _correct_eigenvector_streak(U, U_correct):
        n_components = U.shape[0]
        cosine_similarities_x = jnp.diag(
            jnp.corrcoef(U.T, U_correct, rowvar=False)[n_components:, :n_components]
        )
        x_idx = jnp.where(
            jnp.abs(cosine_similarities_x) > jnp.cos(jnp.pi / 8),
            jnp.ones_like(cosine_similarities_x),
            jnp.zeros_like(cosine_similarities_x),
        )
        return jnp.sum(x_idx)

    @staticmethod
    @jit
    def _normalized_subspace_distance(U, U_correct):
        U = U.T / jnp.linalg.norm(U, axis=1)
        P = U_correct @ U_correct.T
        U_star = U @ U.T
        return 1 - jnp.trace(U_star @ P) / U_correct.shape[1]

    def evaluate(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer,
    ):
        return {}
