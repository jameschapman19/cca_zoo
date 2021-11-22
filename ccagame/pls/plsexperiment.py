from abc import abstractmethod
from typing import Optional
from jax._src.numpy.lax_numpy import zeros_like
import jax.numpy as jnp
from ccagame.baseexperiment import BaseExperiment
from jaxline import utils
from jax import jit
from functools import partial

class PLSExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        batch_size=0,
        correct_eigenvectors=None,
        **kwargs,
    ):
        super(PLSExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=data,
            batch_size=batch_size,
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self.dims = dims
        self.correct_eigenvectors = correct_eigenvectors

    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    #@partial(jit, static_argnums=(0))
    def _get_scalars(self):
        U = jnp.reshape(self._U, (self.n_components, self.dims[0]))
        V = jnp.reshape(self._V, (self.n_components, self.dims[1]))
        return {
            #"TV": self._TV(self., V),
            "correct x":self._correct_eigenvector_streak(U, self.correct_eigenvectors[0]),
            "correct y":self._correct_eigenvector_streak(V, self.correct_eigenvectors[1]),
            "subspace x":self._normalized_subspace_distance(U, self.correct_eigenvectors[0]),
            "subspace y":self._normalized_subspace_distance(V, self.correct_eigenvectors[1]),
        }

    @partial(jit, static_argnums=(0))
    def _TV(self, U, V):
        X, Y = next(self.data_stream)
        X = jnp.reshape(X, (-1, X.shape[-1]))
        Y = jnp.reshape(Y, (-1, X.shape[-1]))
        U = jnp.reshape(U, (-1, X.shape[-1]))
        V = jnp.reshape(V, (-1, X.shape[-1]))
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
