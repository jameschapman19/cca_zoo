from typing import List, Optional

import torch

from cca_zoo.deep._discriminative._dcca_ey import DCCA_EY, _CCA_EYLoss
from cca_zoo.deep._utils import CCA_CV


class DCCA_GHA(DCCA_EY):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dimensions: int, encoders=None, eps: float = 0, **kwargs):
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, eps=eps, **kwargs
        )
        self.objective = _CCA_GHALoss()


class _CCA_GHALoss(_CCA_EYLoss):
    @staticmethod
    @torch.jit.script
    def __call__(
        representations: List[torch.Tensor],
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        C, V = CCA_CV(representations)
        C=C+V
        rewards = torch.trace(C)
        if independent_representations is None:
            rewards.add_(torch.trace(C))
            penalties = torch.trace(C @ V)
        else:
            independent_C, independent_V = CCA_CV(independent_representations)
            independent_C=independent_C+independent_V
            rewards.add_(torch.trace(independent_C))
            penalties = torch.trace(independent_C @ V)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
