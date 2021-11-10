from abc import abstractmethod

import jax.numpy as jnp
from ccagame.baseexperiment import BaseExperiment


class PCAExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        dims=1,
        data_stream=None,
    ):
        super(PCAExperiment, self).__init__(
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
        X = next(self.data_stream)
        _, vals, vecs = jnp.linalg.svd(X.T @ X)
        self._correct_eigenvectors = vecs[: self._total_k, :]
        self._correct_eigenvalues = vals[: self._total_k] / X.shape[0]

    @abstractmethod
    def _update(self, X_i, Y_i, global_step):
        raise NotImplementedError

    def _get_scalars(self, V):
        return {
            # "TV": self.TV(V),
            "Correct Eigenvector Streak": self._correct_eigenvector_streak(V),
        }

    def TV(self, V):
        X = next(self.data_stream)
        X = jnp.reshape(X, (-1, X.shape[-1]))
        V = jnp.reshape(V, (-1, X.shape[-1]))
        dof = X.shape[0]
        U = X @ V.T
        return jnp.linalg.svd(U.T @ U)[1].sum() / dof

    def _correct_eigenvector_streak(self, V):
        V = jnp.reshape(V, (-1, V.shape[-1]))
        cosine_similarities = jnp.diag(self._correct_eigenvectors @ V.T)#V.T@V
        close = jnp.where(jnp.abs(cosine_similarities) < jnp.cos(jnp.pi / 8))[0]
        if len(close) == 0:
            return self._total_k
        else:
            return close[0]
