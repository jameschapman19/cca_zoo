from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from ccagame.baseexperiment import BaseExperiment
from jax import jit


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
        super(PCAExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=data,
            pca=True,
            batch_size=batch_size,
            **kwargs,
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self.TV = TV

    def _init_ground_truth(self, X, Y=None):
        correct_V, _, _ = np.linalg.svd(X.T @ X)
        self.correct_V = correct_V[:, : self.n_components]

    @abstractmethod
    def _update(self, X_i, Y_i, global_step):
        raise NotImplementedError

    def _get_scalars(self):
        scalars = {}
        if self.TV:
            scalars["TV"] = self._TV(self._V, self.X_val)
        scalars["correct_x"] = self._correct_eigenvector_streak(self._V, self.correct_V)
        scalars["subspace"] = self._normalized_subspace_distance(
            self._V, self.correct_V
        )
        return scalars

    @staticmethod
    @jit
    def _TV(U, X_val):
        dof = X_val.shape[0]
        Zx = X_val @ U.T
        return jnp.sum(jnp.linalg.svd(Zx.T @ Zx)[1]) / dof

    def save_outputs(self):
        V = jnp.reshape(self._V, (self.n_components, self.dims))
        np.savetxt("V.csv", V, delimiter=",")
