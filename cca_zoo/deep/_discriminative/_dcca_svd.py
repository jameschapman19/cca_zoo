from typing import List, Optional

import torch

from cca_zoo.deep._discriminative._dcca_ey import DCCA_EY, _CCA_EYLoss


class DCCA_SVD(DCCA_EY):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A GeneralizedDeflation EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dimensions: int, encoders=None, eps: float = 0, **kwargs):
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, eps=eps, **kwargs
        )
        self.objective = _CCA_SVDLoss


class _CCA_SVDLoss(_CCA_EYLoss):
    @staticmethod
    @torch.jit.script
    def __call__(
        representations: List[torch.Tensor],
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        C = torch.cov(torch.hstack(representations).T)
        latent_dims = representations[0].shape[1]

        Cxy = C[:latent_dims, latent_dims:]
        Cxx = C[:latent_dims, :latent_dims]

        if independent_representations is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            Cyy = torch.cov(independent_representations[1].T)

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)
        return {
            "objective": -rewards + penalties,  # return the negative objective value
            "rewards": rewards,  # return the total rewards
            "penalties": penalties,  # return the penalties matrix
        }
