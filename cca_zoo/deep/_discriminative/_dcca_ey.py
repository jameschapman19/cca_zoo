from typing import Dict, Any

import torch

from ._dcca import DCCA


class DCCA_EY(DCCA):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A GeneralizedDeflation EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dimensions: int, encoders=None, r: float = 0, **kwargs):
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, **kwargs
        )
        self.r = r

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs one step of training on a batch of views."""
        loss = self.loss(batch["views"], batch.get("independent_views", None))
        for k, v in loss.items():
            # Use f-string instead of concatenation
            self.log(f"train/{k}", v, prog_bar=True)
        return loss["objective"]

    def loss(self, views, independent_views=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)
        rewards = torch.trace(2 * A)
        if independent_views is None:
            penalties = torch.trace(B @ B)
        else:
            # Encoding another set of views with the forward method
            independent_z = self(independent_views)
            # Getting A' and B' matrices from independent_z
            independent_A, independent_B = self.get_AB(independent_z)
            penalties = torch.trace(B @ independent_B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    def get_AB(self, z):
        A = torch.zeros(
            self.latent_dimensions, self.latent_dimensions, device=z[0].device
        )  # initialize the cross-covariance matrix
        B = torch.zeros(
            self.latent_dimensions, self.latent_dimensions, device=z[0].device
        )  # initialize the auto-covariance matrix
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    B += torch.cov(zi.T)  # add the auto-covariance of each view to B
                else:
                    A += torch.cov(torch.hstack((zi, zj)).T)[
                        self.latent_dimensions :, : self.latent_dimensions
                    ]  # add the cross-covariance of each pair of views to A
        return A / len(z), B / len(
            z
        )  # return the normalized matrices (divided by the number of views)
