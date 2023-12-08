from typing import List, Optional

import torch

from ._dcca import DCCA
from .._utils import CCA_AB


class DCCA_EY(DCCA):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dimensions: int, encoders=None, eps: float = 0, **kwargs):
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, eps=eps, **kwargs
        )
        self.objective = _CCA_EYLoss

    def loss(self, batch, **kwargs):
        # Encoding the representations with the forward method
        representations = self(batch["views"])
        if batch.get("independent_views") is None:
            independent_representations = None
        else:
            independent_representations = self(batch["independent_views"])
        return self.objective(representations, independent_representations)


class _CCA_EYLoss:

    @staticmethod
    @torch.jit.script
    def __call__(
        representations: List[torch.Tensor],
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        A, B = CCA_AB(representations)
        rewards = torch.trace(2 * A)
        if independent_representations is None:
            penalties = torch.trace(B @ B)
        else:
            independent_A, independent_B = CCA_AB(independent_representations)
            penalties = torch.trace(B @ independent_B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
