from abc import abstractmethod
from typing import Optional
import jax.numpy as jnp
from ccagame.baseexperiment import BaseExperiment
from jax._src.random import PRNGKey
from jaxline import utils


class PLSExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        dims=1,
        data_stream=None,
    ):
        super(PLSExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            k_per_device=k_per_device,
            data_stream=data_stream,
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._dims = dims
        X, Y = next(self.data_stream)
        self._correct_U, self._correct_S, self._correct_Vt = jnp.linalg.svd(X.T @ Y)
        self._correct_U = self._correct_U[:, : self._total_k]
        self._correct_Vt = self._correct_Vt[: self._total_k, :]

    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    def _get_scalars(self, ouputs):
        U, V = ouputs
        return {
            # "TV": self.TV(X, V),
            **self._correct_eigenvector_streak(U, V),
        }

    def TV(self, U, V):
        X, Y = next(self.data_stream)
        X = jnp.reshape(X, (-1, X.shape[-1]))
        Y = jnp.reshape(Y, (-1, X.shape[-1]))
        U = jnp.reshape(U, (-1, X.shape[-1]))
        V = jnp.reshape(V, (-1, X.shape[-1]))
        dof = X.shape[0]
        Zx = X @ U.T
        Zy = Y @ V.T
        return jnp.linalg.svd(Zx.T @ Zy)[1].sum() / dof

    def _correct_eigenvector_streak(self, U, V):
        U = jnp.reshape(U, (-1, U.shape[-1]))
        V = jnp.reshape(V, (-1, V.shape[-1]))
        cosine_similarities_x = jnp.diag(self._correct_U.T @ U.T)
        cosine_similarities_y = jnp.diag(self._correct_Vt @ V.T)
        x_idx = jnp.where(jnp.abs(cosine_similarities_x) < jnp.cos(jnp.pi / 8))[0]
        if len(x_idx) == 0:
            x_idx = 0
        else:
            x_idx = x_idx[0]
        y_idx = jnp.where(jnp.abs(cosine_similarities_y) < jnp.cos(jnp.pi / 8))[0]
        if len(y_idx) == 0:
            y_idx = 0
        else:
            y_idx = y_idx[0]
        return {"correct x": x_idx, "correct y": y_idx}

    def evaluate(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        return {}
