"""DCCA_SDL — Stochastic Decorrelation Loss (Chang 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._dcca import DCCA


def _sdl_loss(view: torch.Tensor) -> torch.Tensor:
    """Compute the SDL off-diagonal covariance penalty for a single view.

    Penalises the absolute values of off-diagonal entries of the
    within-view covariance matrix, encouraging feature decorrelation.

    Args:
        view: Tensor of shape (batch_size, latent_dimensions).

    Returns:
        Scalar tensor: mean of |off-diagonal covariance entries|.
    """
    cov = torch.cov(view.T)
    mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
    return cov[mask].abs().mean()


class DCCA_SDL(DCCA):
    """Deep CCA via Stochastic Decorrelation Loss.

    Combines an MSE alignment loss between views with a within-view
    decorrelation penalty.  Batch normalisation is applied to each
    encoder output before the loss is computed.

    The total loss is::

        L = MSE(z1, z2) + lam * (SDL(z1) + SDL(z2))

    where SDL(z) = mean|off-diag(Cov(z))|.

    Reference:
        Chang, X., Xiang, T., & Hospedales, T. M. "Scalable and
        effective deep CCA via soft decorrelation." CVPR 2018.

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        lam: Weight of the SDL decorrelation penalty. Default is 0.5.
        objective: Ignored; the SDL loss is fixed. Accepted for API
            compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Regularisation for numerical stability. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = DCCA_SDL(latent_dimensions=4, encoders=[enc1, enc2], lam=0.5)
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        lam: float = 0.5,
        objective: nn.Module | None = None,
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            objective=objective,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.lam = lam
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(latent_dimensions, affine=False) for _ in encoders]
        )

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode views and apply batch normalisation.

        Args:
            views: List of input tensors, one per view.

        Returns:
            List of batch-normalised latent tensors.
        """
        return [bn(enc(v)) for enc, bn, v in zip(self.encoders, self.bns, views)]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the SDL loss.

        Args:
            representations: Encoded and batch-normalised views from the
                current batch, each of shape (batch_size, latent_dimensions).
            independent_representations: Unused.

        Returns:
            Dictionary with keys ``"objective"``, ``"l2"``, and ``"sdl"``.
        """
        l2 = F.mse_loss(representations[0], representations[1])
        sdl = torch.stack([_sdl_loss(r) for r in representations]).sum()
        return {
            "objective": l2 + self.lam * sdl,
            "l2": l2,
            "sdl": sdl,
        }
