import torch
from ._dcca_eigengame import DCCA_EigenGame


class DCCA_GHA(DCCA_EigenGame):
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
