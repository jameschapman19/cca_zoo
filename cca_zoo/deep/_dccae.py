"""Deep canonically correlated autoencoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import Batch
from cca_zoo.deep._dcca import DCCA


def _reconstruction_loss(
    views: list[torch.Tensor], reconstructions: list[torch.Tensor]
) -> torch.Tensor:
    """Summed mean squared reconstruction error over views."""
    return torch.stack([F.mse_loss(r, x) for x, r in zip(views, reconstructions)]).sum()


class DCCAE(DCCA):
    r"""Deep CCA with per-view autoencoder reconstruction.

    $$
    \mathcal{L} = (1 - \lambda) \mathcal{L}_{\text{CCA}}(z_1, \dots, z_M)
        + \lambda \sum_i \operatorname{MSE}(x_i, \text{decoder}_i(z_i)).
    $$

    ``lam=0`` is :class:`DCCA` and ``lam=1`` independent autoencoders.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        decoders: One module per view mapping its encoding back.
        lam: Reconstruction weight in ``[0, 1]``. Default is 0.5.
        objective: Correlation loss; None uses ``CCALoss(eps)``. Default is
            None.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Ridge of the default loss. Default is 1e-6.

    Raises:
        ValueError: If ``lam`` is outside ``[0, 1]``, or ``objective`` is None
            and there are not two encoders.

    References:
        Wang, W., Arora, R., Livescu, K., & Bilmes, J. (2015). On deep
        multi-view representation learning. ICML.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DCCAE
        >>> model = DCCAE(
        ...     n_components=4,
        ...     encoders=[nn.Linear(10, 4), nn.Linear(8, 4)],
        ...     decoders=[nn.Linear(4, 10), nn.Linear(4, 8)],
        ... )
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        decoders: list[nn.Module],
        lam: float = 0.5,
        objective: nn.Module | None = None,
        learning_rate: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            objective=objective,
            learning_rate=learning_rate,
            eps=eps,
        )
        self.lam = lam
        self.decoders = nn.ModuleList(decoders)

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The DCCAE loss of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "cca", "reconstruction"}``.
        """
        views = batch["views"]
        representations = self(views)
        cca = self.objective(representations)
        reconstruction = _reconstruction_loss(
            views, [dec(z) for dec, z in zip(self.decoders, representations)]
        )
        return {
            "objective": (1.0 - self.lam) * cca + self.lam * reconstruction,
            "cca": cca,
            "reconstruction": reconstruction,
        }
