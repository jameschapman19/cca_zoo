import torch

from ._dcca import DCCA


class DCCA_EigenGame(DCCA):
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
            z.append(encoder(views[i]))
        return z

    def loss(self, views, **kwargs):
        z = self(views)
        A, B = self.get_AB(z)
        rewards = 2 * torch.trace(A)
        penalties = torch.trace(A @ B).sum()
        return {
            "objective": -rewards.sum() + penalties,
            "rewards": rewards.sum(),
            "penalties": penalties,
        }

    def get_AB(self, z):
        # sum the pairwise covariances between each z and all other zs
        A = torch.zeros(self.latent_dims, self.latent_dims)
        B = torch.zeros(self.latent_dims, self.latent_dims)
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    B += (
                        torch.cov(zi.T)
                        + torch.eye(self.latent_dims, device=zi.device) * self.r
                    )
                else:
                    A += torch.cov(torch.hstack((zi, zj)).T)[
                        self.latent_dims :, : self.latent_dims
                    ]
        return A, B
