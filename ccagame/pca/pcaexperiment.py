from abc import abstractmethod

import jax.numpy as jnp
from ccagame.baseexperiment import BaseExperiment


class PCAExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        batch_size=False
    ):
        super(PCAExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=data,
            batch_size=batch_size
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._dims = dims
        X = list(self.data_stream)
        _, vals, vecs = jnp.linalg.svd(X.T @ X)
        self._correct_eigenvectors = vecs[: self.n_components, :]
        self._correct_eigenvalues = vals[: self.n_components] / X.shape[0]
        if self.batch_size:
            self.inputs=next(self.data_stream)

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
            return self.n_components
        else:
            return close[0]
