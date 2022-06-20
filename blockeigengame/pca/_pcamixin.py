from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from jax import jit

from .._baseexperiment import _BaseExperiment


class _PCAMixin:

    def _init_ground_truth(self, X, Y=None):
        correct_V, _, _ = np.linalg.svd(X.T @ X)
        self.correct_V = correct_V[:, : self.n_components]

    def _get_scalars(self, global_step):
        scalars = {}
        if global_step == 0 or (global_step + 1) % self.val_interval == 0:
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
