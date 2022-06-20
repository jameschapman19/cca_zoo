from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from cca_zoo.models import PLS

from blockeigengame.datasets.xrmb import xrmb_true
from blockeigengame.metrics import (
    _correct_eigenvector_streak,
    _normalized_subspace_distance,
    _sum_cosine_similarities,
)
from .._baseexperiment import _BaseExperiment
from jax import jit


class _PLSMixin:
    def _init_ground_truth(self, X, Y):
        if self.data == "xrmb":
            self.correct_U, self.correct_V = xrmb_true()
            self.correct_U = self.correct_U[:, : self.config.n_components]
            self.correct_V = self.correct_V[:, : self.config.n_components]
        else:
            U, _, Vt = jnp.linalg.svd(X.T @ Y)
            self.correct_U = U[:, : self.config.n_components]
            self.correct_V = Vt[: self.config.n_components, :].T
        if self.TV:
            if self.data != "xrmb":
                self.TV_train = self._TV(
                    self.correct_U.T, self.correct_V.T, self.X, self.Y
                )
            self.TV_val = self._TV(
                self.correct_U.T, self.correct_V.T, self.X_val, self.Y_val
            )

    def _get_scalars(self, global_step):
        scalars = {}
        if global_step == 0 or (global_step + 1) % self.val_interval == 0:
            if self.TV:
                if self.data != "xrmb":
                    scalars["TV train"] = _TV(self._U, self._V, self.X, self.Y)
                    scalars["PV train"] = scalars["TV train"] / self.TV_train
                scalars["TV val"] = _TV(self._U, self._V, self.X_val, self.Y_val)
                scalars["PV val"] = scalars["TV val"] / self.TV_val
            scalars["correct x"] = _correct_eigenvector_streak(self._U, self.correct_U)
            scalars["correct y"] = _correct_eigenvector_streak(self._V, self.correct_V)
            scalars["sum cosine similarities x"] = _sum_cosine_similarities(
                self._U, self.correct_U
            )
            scalars["sum cosine similarities y"] = _sum_cosine_similarities(
                self._V, self.correct_V
            )
            scalars["subspace x"] = _normalized_subspace_distance(
                self._U, self.correct_U
            )
            scalars["subspace y"] = _normalized_subspace_distance(
                self._V, self.correct_V
            )
        return scalars


@staticmethod
@jit
def _TV(U, V, X_val, Y_val):
    dof = X_val.shape[0]
    Qu, Ru = jnp.linalg.qr(U.T)
    Su = jnp.sign(jnp.sign(jnp.diag(Ru)) + 0.5)
    Qv, Rv = jnp.linalg.qr(V.T)
    Sv = jnp.sign(jnp.sign(jnp.diag(Rv)) + 0.5)
    return (
        jnp.trace(
            jnp.atleast_2d(
                (Qu @ jnp.diag(Su)).T @ X_val.T @ Y_val @ (Qv @ jnp.diag(Sv))
            )
        )
        / dof
    )
