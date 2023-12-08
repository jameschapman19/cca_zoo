from typing import List, Optional

import torch

from ._dcca import DCCA
from .._utils import CCA_CV

class _CCA_EYLoss:

    @staticmethod
    @torch.jit.script
    def __call__(
        representations: List[torch.Tensor],
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        C, V = CCA_CV(representations)
        rewards = torch.trace(2 * C)
        if independent_representations is None:
            penalties = torch.trace(V @ V)
        else:
            independent_C, independent_V = CCA_CV(independent_representations)
            penalties = torch.trace(V @ independent_V)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }


class DCCA_EY(DCCA):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """
    objective = _CCA_EYLoss()

    def loss(self, batch, **kwargs):
        # Encoding the representations with the forward method
        representations = self(batch["views"])
        if batch.get("independent_views") is None:
            independent_representations = None
        else:
            independent_representations = self(batch["independent_views"])
        return self.objective(representations, independent_representations)