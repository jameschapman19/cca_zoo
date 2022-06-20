from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from cca_zoo.models import PLS

from blockeigengame.datasets.xrmb import xrmb_true
from .._baseexperiment import _BaseExperiment
from jax import jit

class _PLSMixin:
    def _init_ground_truth(self, X, Y):
        if self.data=='xrmb':
            self.correct_U,self.correct_V=xrmb_true()
            self.correct_U=self.correct_U[:,:self.n_components]
            self.correct_V=self.correct_V[:,:self.n_components]
        else:
            U, _, Vt = jnp.linalg.svd(X.T @ Y)
            self.correct_U = U[
                :, : self.n_components
            ]
            self.correct_V = Vt[: self.n_components, :].T
        if self.TV:
            if self.data!='xrmb':
                self.TV_train = self._TV(self.correct_U.T, self.correct_V.T, self.X, self.Y)
            self.TV_val = self._TV(
                self.correct_U.T, self.correct_V.T, self.X_val, self.Y_val
            )

    def _get_scalars(self, global_step):
        scalars = {}
        if global_step == 0 or (global_step + 1) % self.val_interval == 0:
            if self.TV:
                if self.data!='xrmb':
                    scalars["TV train"] = self._TV(self._U, self._V, self.X, self.Y)
                    scalars["PV train"] = scalars["TV train"] / self.TV_train
                scalars["TV val"] = self._TV(self._U, self._V, self.X_val, self.Y_val)
                scalars["PV val"] = scalars["TV val"] / self.TV_val
            scalars["correct x"] = self._correct_eigenvector_streak(
                self._U, self.correct_U
            )
            scalars["correct y"] = self._correct_eigenvector_streak(
                self._V, self.correct_V
            )
            scalars["sum cosine similarities x"] = self._sum_cosine_similarities(
                self._U, self.correct_U
            )
            scalars["sum cosine similarities y"] = self._sum_cosine_similarities(
                self._V, self.correct_V
            )
            scalars["subspace x"] = self._normalized_subspace_distance(
                self._U, self.correct_U
            )
            scalars["subspace y"] = self._normalized_subspace_distance(
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
        return jnp.trace(jnp.atleast_2d((Qu @ jnp.diag(Su)).T @ X_val.T @ Y_val @ (Qv @ jnp.diag(Sv)))) / dof

    def save_outputs(self):
        np.savetxt("U.csv", self._U, delimiter=",")
        np.savetxt("V.csv", self._V, delimiter=",")

    @staticmethod
    @jit
    def _sum_cosine_similarities(U, U_correct):
        n_components = U.shape[0]
        cosine_similarities = jnp.diag(
            jnp.corrcoef(U.T, U_correct, rowvar=False)[n_components:, :n_components]
        )
        return jnp.sum(jnp.abs(cosine_similarities))
