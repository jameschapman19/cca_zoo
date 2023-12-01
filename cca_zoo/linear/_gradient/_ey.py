from cca_zoo._utils._cross_correlation import cross_cov
from cca_zoo.linear._gradient._base import BaseGradientModel
from cca_zoo.linear._gradient._objectives import CCA_AB, PLS_AB
import numpy as np
from typing import List, Optional


class CCA_EY(BaseGradientModel):
    def loss(
        self,
        representations: List[np.ndarray],
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        A, B = CCA_AB(representations)
        rewards = np.trace(2 * A)
        if independent_representations is None:
            penalties = np.trace(B @ B)
        else:
            independent_A, independent_B = CCA_AB(independent_representations)
            penalties = np.trace(B @ independent_B)
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
        rewards = [2 * view.T @ sum_representations / (n - 1) for view in views]
        if independent_representations is None:
            penalties = [
                2 * view.T @ representation @ B / (n - 1)
                for view, representation in zip(views, representations)
            ]
        else:
            _, independent_B = CCA_AB(independent_representations)
            penalties = [
                view.T @ representation @ B / (n - 1)
                + independent_view.T
                @ independent_representation
                @ independent_B
                / (n - 1)
                for view, representation, independent_view, independent_representation in zip(
                    views,
                    representations,
                    independent_views,
                    independent_representations,
                )
            ]
        return [2 * (-reward + penalty) for reward, penalty in zip(rewards, penalties)]


class PLS_EY(BaseGradientModel):
    def loss(
        self,
        representations: List[np.ndarray],
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        A, B = PLS_AB(representations, self.weights_)
        rewards = np.trace(2 * A)
        penalties = np.trace(B @ B)
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
        sum_representations = np.sum(np.stack(representations), axis=0)
        rewards = [
            2 * cross_cov(view, sum_representations, rowvar=False)
            - 2 * cross_cov(view, representation, rowvar=False)
            for view, representation in zip(views, representations)
        ]
        penalties = [
            2 * weights @ (weights.T @ weights) / (view.shape[0] - 1)
            for view, representation, weights in zip(
                views, representations, self.weights_
            )
        ]
        return [2 * (-reward + penalty) for reward, penalty in zip(rewards, penalties)]
