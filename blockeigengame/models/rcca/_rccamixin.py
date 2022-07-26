import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from cca_zoo.models import MCCA
from jax import jit

from blockeigengame.data_utils.mediamill import mediamill_true
from blockeigengame.data_utils.xrmb import xrmb_true
from blockeigengame.metrics import _correct_eigenvector_streak, _sum_cosine_similarities


class _RCCAMixin:
    def _init_ground_truth(self):
        if self.config.data == "xrmb":
            self.correct_U, self.correct_V = xrmb_true(cca=True)
            self.correct_U = self.correct_U[:, : self.config.n_components]
            self.correct_V = self.correct_V[:, : self.config.n_components]
        elif self.config.data == "mediamill":
            self.correct_U, self.correct_V = mediamill_true(cca=True)
            self.correct_U = self.correct_U[:, : self.config.n_components]
            self.correct_V = self.correct_V[:, : self.config.n_components]
        else:
            cca = MCCA(latent_dims=self.config.n_components).fit((self.X, self.Y))
            self.correct_U, self.correct_V = cca.weights
        self.TCC_train = _TCC(self.X, self.Y, self.correct_U.T, self.correct_V.T)
        self.TCC_val = _TCC(self.X_val, self.Y_val, self.correct_U.T, self.correct_V.T)

    def _get_scalars(self, global_step):
        scalars = {}
        if global_step == 0 or (global_step + 1) % self.config.val_interval == 0:
            scalars["TCC train"] = _TCC(self.X, self.Y, self._U, self._V)
            scalars["TCC val"] = _TCC(self.X_val, self.Y_val, self._U, self._V)
            scalars["PCC train"] = scalars["TCC train"] / self.TCC_train
            scalars["PCC val"] = scalars["TCC val"] / self.TCC_val
            scalars["correct x"] = _correct_eigenvector_streak(self._U, self.correct_U)
            scalars["correct y"] = _correct_eigenvector_streak(self._V, self.correct_V)
            scalars["sum cosine similarities x"] = _sum_cosine_similarities(
                self._U, self.correct_U
            )
            scalars["sum cosine similarities y"] = _sum_cosine_similarities(
                self._V, self.correct_V
            )
            return scalars

    def evaluate(self, global_step, rng):
        scalars = {}
        scalars["TCC train"] = _TCC(self.X, self.Y, self._U, self._V)
        scalars["TCC val"] = _TCC(self.X_val, self.Y_val, self._U, self._V)
        scalars["PCC train"] = scalars["TCC train"] / self.TCC_train
        scalars["PCC val"] = scalars["TCC val"] / self.TCC_val
        scalars["correct x"] = _correct_eigenvector_streak(self._U, self.correct_U)
        scalars["correct y"] = _correct_eigenvector_streak(self._V, self.correct_V)
        scalars["sum cosine similarities x"] = _sum_cosine_similarities(
            self._U, self.correct_U
        )
        scalars["sum cosine similarities y"] = _sum_cosine_similarities(
            self._V, self.correct_V
        )
        return scalars


@jit
def _TCC(X, Y, U, V):
    Zx = X @ U.T
    Zy = Y @ V.T
    all = jnp.hstack((Zx, Zy))
    C = all.T @ all
    D = jsp.linalg.block_diag(Zx.T @ Zx, Zy.T @ Zy)
    D = D + 1e-3 * jnp.eye(C.shape[0])
    C = jnp.linalg.pinv(D) @ C
    try:
        return (jsp.linalg.eigh(C)[0] - 1)[-U.shape[0]:].sum()
    except:
        return np.nan
