import torch

from ._dcca import DCCA


class DCCA_EigenGame(DCCA):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(
            self,
            latent_dims: int,
            encoders=None,
            r: float = 0,
            **kwargs
    ):
        super().__init__(
            latent_dims=latent_dims,
            encoders=encoders,
            **kwargs
        )
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
        Cxy = torch.cov(torch.hstack((z[0], z[1])).T)[self.latent_dims:, :self.latent_dims]
        Cxx = torch.cov(z[0].T) + torch.eye(self.latent_dims, device=z[0].device) * self.r
        Cyy = torch.cov(z[1].T) + torch.eye(self.latent_dims, device=z[1].device) * self.r
        return Cxy + Cxy.T, Cxx + Cyy
