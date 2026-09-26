"""DCCASDL — Stochastic Decorrelation Loss (Chang 2018)."""

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
        view: Tensor of shape (batch_size, n_components).

    Returns:
        Scalar tensor: mean of |off-diagonal covariance entries|.
    """
    cov = torch.cov(view.T)
    mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
    return cov[mask].abs().mean()


class DCCASDL(DCCA):
    r"""Deep CCA via Stochastic Decorrelation Loss.

    Combines an MSE alignment loss between views with a within-view
    decorrelation penalty.  Batch normalisation is applied to each
    encoder output before the loss is computed.

    $$
    \mathcal{L} = \operatorname{MSE}(z_1, z_2)
        + \lambda \sum_{i} \operatorname{SDL}(z_i), \qquad
    \operatorname{SDL}(z) = \operatorname{mean}\bigl(
        \left|\text{off-diag}(\operatorname{Cov}(z))\right|
    \bigr)
    $$

    References:
        Chang, X., Xiang, T., & Hospedales, T. M. "Scalable and
        effective deep CCA via soft decorrelation." CVPR 2018.

    Args:
        n_components: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        lam: Weight of the SDL decorrelation penalty. Default is 0.5.
        learning_rate: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = DCCASDL(n_components=4, encoders=[enc1, enc2], lam=0.5)
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        lam: float = 0.5,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
        )
        self.lam = lam
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(n_components, affine=False) for _ in encoders]
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
                current batch, each of shape (batch_size, n_components).
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
