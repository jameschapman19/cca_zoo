import torch

from ._dcca_ey import DCCA_EY


class DCCA_GH(DCCA_EY):
    def loss(self, views, **kwargs):
        z = self(views)
        A, B = self.get_AB(z)
        rewards = torch.trace(A) + torch.trace(A).detach()
        penalties = torch.trace(A.detach() @ B)
        return {
            "objective": -rewards.sum() + penalties,
            "rewards": rewards.sum(),
            "penalties": penalties,
        }
