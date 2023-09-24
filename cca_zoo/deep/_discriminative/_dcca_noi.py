import torch

from ._dcca import DCCA
from ..objectives import inv_sqrtm


class DCCA_NOI(DCCA):
    """
    A class used to fit a DCCA model by non-linear orthogonal iterations


    References
    ----------
    Wang, Weiran, et al. "Stochastic optimization for deep CCA via nonlinear orthogonal iterations." 2015 53rd Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2015.

    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders=None,
        r: float = 0,
        rho: float = 0.2,
        eps: float = 1e-9,
        **kwargs,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            r=r,
            eps=eps,
            **kwargs,
        )
        if rho < 0 or rho > 1:
            raise ValueError(f"rho should be between 0 and 1. rho={rho}")
        self.eps = eps
        self.rho = rho
        self.mse = torch.nn.MSELoss(reduction="sum")

    def on_fit_start(self) -> None:
        self.covs = [
            torch.eye(self.latent_dimensions, requires_grad=False, device=self.device)
        ] * 2

    def on_train_batch_start(self, batch, batch_idx):
        z = self(batch["views"])
        self._update_covariances(z)

    def loss(self, batch, **kwargs):
        z = self(batch["views"])
        covariance_inv = [inv_sqrtm(cov, self.eps) for cov in self.covs]
        preds = [z_ @ covariance_inv[i] for i, z_ in enumerate(z)]
        loss = self.mse(z[0], preds[1].detach()) + self.mse(z[1], preds[0].detach())
        return {"objective": loss}

    def _update_covariances(self, z):
        batch_covs = [torch.cov(z_.T) for z_ in z]
        self.covs = [
            self.rho * self.covs[i] + (1 - self.rho) * batch_cov
            for i, batch_cov in enumerate(batch_covs)
        ]
