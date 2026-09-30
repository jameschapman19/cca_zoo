"""Deep CCA with a stochastic decorrelation loss."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import BaseDeep, Batch


def _sdl_loss(view: torch.Tensor) -> torch.Tensor:
    """Mean absolute off-diagonal covariance of one view's encoding."""
    cov = torch.cov(view.T)
    mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
    return cov[mask].abs().mean()


class DCCASDL(BaseDeep):
    r"""Deep CCA with a stochastic decorrelation loss.

    Batch-normalised encodings are aligned by mean squared error, summed over
    pairs of views, and decorrelated within each view:

    $$
    \mathcal{L} = \sum_{a < b} \operatorname{MSE}(z_a, z_b) + \lambda \sum_a
        \operatorname{mean}\lvert \text{offdiag}(\operatorname{Cov}(z_a)) \rvert.
    $$

    With two views this is the original loss, with one difference: Chang et
    al. estimate the covariance by a running average across batches, while
    here each batch's covariance is used, so no state is carried between
    batches, as in :class:`DPCCA`. Use batches well above ``n_components``.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        lam: Weight of the decorrelation term. Default is 0.5.
        learning_rate: Adam learning rate. Default is 1e-3.

    References:
        Chang, X., Xiang, T., & Hospedales, T. M. (2018). Scalable and
        effective deep CCA via soft decorrelation. CVPR.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DCCASDL
        >>> encoders = [nn.Linear(10, 4), nn.Linear(8, 4)]
        >>> model = DCCASDL(n_components=4, encoders=encoders)
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        lam: float = 0.5,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.lam = lam
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(n_components, affine=False) for _ in encoders]
        )

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch-normalised encodings of each view."""
        return [bn(enc(v)) for enc, bn, v in zip(self.encoders, self.bns, views)]

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The SDL loss of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "l2", "sdl"}``.
        """
        representations = self(batch["views"])
        l2 = torch.stack(
            [F.mse_loss(a, b) for a, b in itertools.combinations(representations, 2)]
        ).sum()
        sdl = torch.stack([_sdl_loss(r) for r in representations]).sum()
        return {
            "objective": l2 + self.lam * sdl,
            "l2": l2,
            "sdl": sdl,
        }
