from typing import List, Optional

import numpy as np

from cca_zoo.linear._gradient._ey import CCA_EY


class CCA_SVD(CCA_EY):
    def loss(
        self,
        representations: List[np.ndarray],
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        C = np.cov(np.hstack(representations).T)
        latent_dims = representations[0].shape[1]

        Cxy = C[:latent_dims, latent_dims:]
        Cxx = C[:latent_dims, :latent_dims]

        if independent_representations is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            Cyy = np.cov(independent_representations[1].T)

        rewards = np.trace(2 * Cxy)
        penalties = np.trace(Cxx @ Cyy)
        return {
            "objective": -rewards + penalties,  # return the negative objective value
            "rewards": rewards,  # return the total rewards
            "penalties": penalties,  # return the penalties matrix
        }

    def derivative(
        self,
        views: List[np.ndarray],
        representations: List[np.ndarray],
        independent_views: Optional[List[np.ndarray]] = None,
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        C = np.cov(np.hstack(representations).T)
        latent_dims = representations[0].shape[1]

        Cxx = C[:latent_dims, :latent_dims]
        sum_representations = np.sum(np.stack(representations), axis=0)
        if independent_representations is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            Cyy = np.cov(independent_representations[1].T)
        n = sum_representations.shape[0]
        rewards = [
            2 * views[0].T @ representations[1] / (n - 1),
            2 * views[1].T @ representations[0] / (n - 1),
        ]
        penalties = [
            views[0].T @ representations[0] @ Cyy / (n - 1),
            views[1].T @ representations[1] @ Cxx / (n - 1),
        ]
        return [2 * (-reward + penalty) for reward, penalty in zip(rewards, penalties)]
