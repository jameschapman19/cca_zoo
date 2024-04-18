from typing import List

import torch

from ._dcca import DCCA
from .._utils import CCA_CV


class DCCA_EY(DCCA):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def loss(
        self,
        representations: List[torch.Tensor],
        independent_representations: List[torch.Tensor] = None,
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
