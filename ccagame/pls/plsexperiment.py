from abc import abstractmethod
from typing import Optional
import jax.numpy as jnp
from ccagame.baseexperiment import BaseExperiment
from jaxline import utils


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
        correct_eigenvectors=None
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

    def _get_scalars(self, ouputs):
        U, V = ouputs
        return {
            # "TV": self.TV(X, V),
            **self._correct_eigenvector_streak(U, V),
            **self._normalized_subspace_distance(U,V),
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
        cosine_similarities_x = jnp.diag(self.correct_eigenvectors[0].T @ U.T)
        cosine_similarities_y = jnp.diag(self.correct_eigenvectors[1].T @ V.T)
        x_idx = jnp.where(jnp.abs(cosine_similarities_x) < jnp.cos(jnp.pi / 8))[0]
        if len(x_idx) == 0:
            x_idx = self.n_components
        else:
            x_idx = x_idx[0]
        y_idx = jnp.where(jnp.abs(cosine_similarities_y) < jnp.cos(jnp.pi / 8))[0]
        if len(y_idx) == 0:
            y_idx = self.n_components
        else:
            y_idx = y_idx[0]
        return {"correct x": x_idx, "correct y": y_idx}

    def _normalized_subspace_distance(self,U,V):
        subspace_distances={}
        P=self.correct_eigenvectors[0] @ self.correct_eigenvectors[0].T
        U_star=U.T @ U
        subspace_distances['x_distance']=1-jnp.trace(U_star@P)/self.n_components
        P=self.correct_eigenvectors[1] @ self.correct_eigenvectors[1].T
        U_star=V.T @ V
        subspace_distances['y_distance']=1-jnp.trace(U_star@P)/self.n_components
        return subspace_distances

    def evaluate(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer: Optional[utils.Writer],
    ):
        return {}
