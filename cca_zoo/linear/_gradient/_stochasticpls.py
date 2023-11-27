from typing import List, Optional

import numpy as np

from cca_zoo.linear._gradient._ey import PLS_EY
from cca_zoo.linear._pls import PLSMixin


class PLSStochasticPower(PLS_EY, PLSMixin):
    def loss(
        self,
        representations: List[np.ndarray],
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        cov = np.cov(np.hstack(representations).T)
        return {
            "objective": np.trace(
                cov[: representations[0].shape[1], representations[0].shape[1] :]
            )
        }

    def derivative(
        self,
        views: List[np.ndarray],
        representations: List[np.ndarray],
        independent_views: Optional[List[np.ndarray]] = None,
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        grads = [views[0].T @ representations[1], views[1].T @ representations[0]]
        return grads

    def on_training_step_start(self):
        self.weights_ = [self._orth(weights) for weights in self.weights_]

    @staticmethod
    def _orth(U):
        Qu, Ru = np.linalg.qr(U)
        Su = np.sign(np.sign(np.diag(Ru)) + 0.5)
        return Qu @ np.diag(Su)
