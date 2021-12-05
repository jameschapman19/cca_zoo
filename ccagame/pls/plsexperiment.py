from abc import abstractmethod
from typing import Optional
import jax.numpy as jnp
from ccagame.baseexperiment import BaseExperiment
from jaxline import utils
from jax import jit
from functools import partial

from datasets.mnist import mnist_iterator
from datasets.xrmb import xrmb_iterator
from datasets.ukbiobank import ukbb_iterator
class PLSExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
        path=None,
        **kwargs,
    ):
        if data=='mnist':
            self.data,self.holdout, self.correct_eigenvectors, self.dims=mnist_iterator(batch_size=batch_size, n_components=n_components)
        elif data=='xrmb':
            self.data,self.holdout, self.correct_eigenvectors, self.dims=xrmb_iterator(batch_size=batch_size, n_components=n_components)
        elif data=='ukbb':
            self.data,self.holdout, self.dims=ukbb_iterator(path, batch_size=batch_size)
            self.correct_eigenvectors = None
        else:
            raise ValueError('Data {data} not implemented yet')
        super(PLSExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=self.data,
            batch_size=batch_size,
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""


    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    #@partial(jit, static_argnums=(0))
    def _get_scalars(self):
        U = jnp.reshape(self._U, (self.n_components, self.dims[0]))#self.correct_eigenvectors[0].T @ U.T
        V = jnp.reshape(self._V, (self.n_components, self.dims[1]))
        if self.correct_eigenvectors == None:
            return {"TV": self._TV(U,V)}
        else:
            return {
                "TV": self._TV(U,V),
                "correct x":self._correct_eigenvector_streak(U, self.correct_eigenvectors[0]),
                "correct y":self._correct_eigenvector_streak(V, self.correct_eigenvectors[1]),
                "subspace x":self._normalized_subspace_distance(U, self.correct_eigenvectors[0]),
                "subspace y":self._normalized_subspace_distance(V, self.correct_eigenvectors[1]),
            }

    @partial(jit, static_argnums=(0))
    def _TV(self, U, V):
        if self.holdout is None:
            return 0
        else:
            X, Y = self.holdout
            dof = X.shape[0]
            Zx = X @ U.T
            Zy = Y @ V.T
            return jnp.linalg.svd(Zx.T @ Zy)[1].sum() / dof

    @staticmethod
    @jit
    def _correct_eigenvector_streak(U, U_correct):
        cosine_similarities_x = jnp.diag(U_correct.T @ U.T)
        x_idx = jnp.where(jnp.abs(cosine_similarities_x) > jnp.cos(jnp.pi / 8),jnp.ones_like(cosine_similarities_x),jnp.zeros_like(cosine_similarities_x))
        return x_idx.sum()

    @staticmethod
    @jit
    def _normalized_subspace_distance(U, U_correct):
        P = U_correct @ U_correct.T
        U_star = U.T @ U
        return 1 - jnp.trace(U_star @ P) / U_correct.shape[1]

    def evaluate(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        return {}
