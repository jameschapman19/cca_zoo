import torch

from ._dcca_ey import DCCA_EY


class DCCA_GH(DCCA_EY):
    def get_AB(self, z):
        A = torch.zeros(
            self.latent_dims, self.latent_dims, device=z[0].device
        )  # initialize the cross-covariance matrix
        B = torch.zeros(
            self.latent_dims, self.latent_dims, device=z[0].device
        )  # initialize the auto-covariance matrix
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    B += torch.cov(zi.T)  # add the auto-covariance of each view to B
                A += torch.cov(torch.hstack((zi, zj)).T)[
                    self.latent_dims :, : self.latent_dims
                ]  # add the cross-covariance of each pair of views to A
        return A / len(z), B / len(
            z
        )  # return the normalized matrices (divided by the number of views)

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
