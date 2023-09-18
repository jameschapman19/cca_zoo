import torch

from cca_zoo.deep._discriminative._dcca_ey import DCCA_EY


class DCCA_SVD(DCCA_EY):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dimensions: int, encoders=None, r: float = 0, **kwargs):
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, r=r, **kwargs
        )
        self.r = r
        # check if the number of views is equal to 2
        if len(self.encoders) != 2:
            raise ValueError(
                f"Expected 2 views, got {len(self.encoders)} views instead."
            )

    def loss(self, views, independent_views=None, **kwargs):
        # views here is a list of 'paired' views (i.e. [view1, view2])
        z = self(views)  # get the latent representations
        C = torch.cov(torch.hstack(z).T)
        latent_dims = z[0].shape[1]

        Cxy = C[:latent_dims, latent_dims:]
        Cxx = C[:latent_dims, :latent_dims]

        if independent_views is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            independent_z = self(independent_views)
            Cyy = torch.cov(torch.hstack(independent_z).T)[latent_dims:, latent_dims:]

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)
        return {
            "objective": -rewards + penalties,  # return the negative objective value
            "rewards": rewards,  # return the total rewards
            "penalties": penalties,  # return the penalties matrix
        }
