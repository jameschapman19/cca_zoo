import torch

from ._dcca import DCCA


class DCCA_EY(DCCA):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dims: int, encoders=None, r: float = 0, **kwargs):
        super().__init__(latent_dims=latent_dims, encoders=encoders, **kwargs)
        self.r = r

    def forward(self, views, **kwargs):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(views[i]))  # encode each view into a latent representation
        return z  # return a list of latent representations

    def loss(self, views, **kwargs):
        # views here is a list of 'paired' views (i.e. [view1, view2])
        z = self(views)  # get the latent representations
        A, B = self.get_AB(z)  # get the cross-covariance and auto-covariance matrices
        rewards = 2 * torch.trace(
            A
        )  # compute the rewards as the sum of cross-covariances
        penalties = torch.trace(
            B @ B
        )  # compute the penalties as the squared Frobenius norm of auto-covariances
        return {
            "objective": -rewards + penalties,  # return the negative objective value
            "rewards": rewards,  # return the total rewards
            "penalties": penalties,  # return the penalties matrix
        }

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
                else:
                    A += torch.cov(torch.hstack((zi, zj)).T)[
                        self.latent_dims :, : self.latent_dims
                    ]  # add the cross-covariance of each pair of views to A
        return A / len(z), B / len(
            z
        )  # return the normalized matrices (divided by the number of views)
