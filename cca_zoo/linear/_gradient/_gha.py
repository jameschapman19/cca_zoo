from typing import List, Optional

import numpy as np

from cca_zoo.linear._gradient._ey import CCA_EY
from cca_zoo.linear._gradient._objectives import CCA_AB


class CCA_GHA(CCA_EY):
    def loss(
        self,
        representations: List[np.ndarray],
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        A, B = CCA_AB(representations)
        rewards = np.trace(A)
        if independent_representations is None:
            rewards += np.trace(A)
            penalties = np.trace(A @ B)
        else:
            independent_A, independent_B = CCA_AB(independent_representations)
            rewards += np.trace(independent_A)
            penalties = np.trace(independent_A @ B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    def derivative(
        self,
        views: List[np.ndarray],
        representations: List[np.ndarray],
        independent_views: Optional[List[np.ndarray]] = None,
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        A, B = CCA_AB(representations)
        sum_representations = np.sum(np.stack(representations), axis=0)
        n = sum_representations.shape[0]
        if independent_representations is None:
            rewards = [2 * view.T @ sum_representations / (n - 1) for view in views]
            penalties = [
                2 * view.T @ representation @ A / (n - 1)
                for view, representation in zip(views, representations)
            ]
        else:
            independent_A, independent_B = CCA_AB(independent_representations)
            rewards = [2 * view.T @ sum_representations / (n - 1) for view in views]
            penalties = [
                2 * view.T @ representation @ independent_A / (n - 1)
                for view, representation in zip(views, representations)
            ]
        return [2 * (-reward + penalty) for reward, penalty in zip(rewards, penalties)]
