import numpy as np
from cca_zoo.models import CCA
from jax import jit

from blockeigengame.metrics import _correct_eigenvector_streak, _sum_cosine_similarities

from ._utils import _TCC


class _CCAMixin:
    def _init_ground_truth(self, X, Y):
        cca = CCA(latent_dims=self.config.n_components).fit((X, Y))
        self.correct_U, self.correct_V = cca.weights
        self.correct_Zx, self.correct_Zy = cca.transform((self.X_val, self.Y_val))
        if self.TCC:
            self.TCC_train = _TCC(self.X, self.Y, self.correct_U.T, self.correct_V.T)
            self.TCC_val = _TCC(
                self.X_val, self.Y_val, self.correct_U.T, self.correct_V.T
            )

    def _get_scalars(self, global_step):
        scalars = {}  # jnp.corrcoef(self.X@self._U.T,self.Y@self._V.T,rowvar=False)
        if global_step == 0 or (global_step + 1) % self.val_interval == 0:
            if self.TCC:
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
