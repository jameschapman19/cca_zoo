from functools import partial

import jax.numpy as jnp
from jax import jit

from blockeigengame.data_utils.mediamill import mediamill_true
from blockeigengame.data_utils.xrmb import xrmb_true
from blockeigengame.metrics import _correct_eigenvector_streak, _sum_cosine_similarities, _normalized_subspace_distance


class _PLSMixin:
    def _init_ground_truth(self):
        if self.config.data == "xrmb":
            self.correct_U, self.correct_V = xrmb_true()
            self.correct_U = self.correct_U[:, : self.config.n_components]
            self.correct_V = self.correct_V[:, : self.config.n_components]
        elif self.config.data == "mediamill":
            self.correct_U, self.correct_V = mediamill_true()
            self.correct_U = self.correct_U[:, : self.config.n_components]
            self.correct_V = self.correct_V[:, : self.config.n_components]
        else:
            U, _, Vt = jnp.linalg.svd(self.X.T @ self.Y)
            self.correct_U = U[:, : self.config.n_components]
            self.correct_V = Vt[: self.config.n_components, :].T
        self.TV_train = _TV(self.correct_U.T, self.correct_V.T, self.X, self.Y)
        self.TV_val = _TV(self.correct_U.T, self.correct_V.T, self.X_val, self.Y_val)

    def _get_scalars(self, global_step):
        scalars = {}
        scalars["examples"] = (global_step[0] + 1) * self.config.batch_size
        scalars["TV train"] = _TV(self._U, self._V, self.X, self.Y)
        scalars["PV train"] = scalars["TV train"] / self.TV_train
        scalars["TV val"] = _TV(self._U, self._V, self.X_val, self.Y_val)
        scalars["PV val"] = scalars["TV val"] / self.TV_val
        scalars["correct x"] = _correct_eigenvector_streak(self._U, self.correct_U)
        scalars["correct y"] = _correct_eigenvector_streak(self._V, self.correct_V)
        scalars["subspace distance x"] = _normalized_subspace_distance(self._U, self.correct_U)
        scalars["subspace distance y"] = _normalized_subspace_distance(self._V, self.correct_V)
        scalars["sum cosine similarities x"] = _sum_cosine_similarities(
            self._U, self.correct_U
        )
        scalars["sum cosine similarities y"] = _sum_cosine_similarities(
            self._V, self.correct_V
        )
        return scalars


@partial(jit, backend="cpu")
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
