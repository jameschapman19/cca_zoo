"""Deep canonically correlated autoencoders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import BaseDeep
from cca_zoo.deep.objectives import CCALoss


class DCCAE(BaseDeep):
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
        max_epochs: Maximum training epochs. Default is 100.
        eps: Ridge of the default loss. Default is 1e-6.

    Raises:
        ValueError: If ``lam`` is outside ``[0, 1]``.

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
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        if lam < 0.0 or lam > 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
        )
        self.eps = eps
        self.lam = lam
        self.decoders = nn.ModuleList(decoders)
        self.objective: nn.Module = CCALoss(eps=eps) if objective is None else objective

    def _decode(self, representations: list[torch.Tensor]) -> list[torch.Tensor]:
        """Reconstruct each view from its own encoding."""
        return [dec(z) for dec, z in zip(self.decoders, representations)]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """The correlation term alone, since reconstruction needs the inputs.

        Args:
            representations: One encoded tensor per view.
            independent_representations: Unused.

        Returns:
            ``{"objective": correlation loss}``.
        """
        return {"objective": self.objective(representations)}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Full loss of a batch, reconstruction included.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.
            batch_idx: Unused.

        Returns:
            The loss.
        """
        views = batch["views"]
        representations = self(views)
        reconstructions = self._decode(representations)

        cca_loss = self.objective(representations)
        recon_loss = torch.stack(
            [F.mse_loss(x, r) for x, r in zip(views, reconstructions)]
        ).sum()
        objective = (1.0 - self.lam) * cca_loss + self.lam * recon_loss

        loss_dict = {
            "objective": objective,
            "cca": cca_loss,
            "reconstruction": recon_loss,
        }
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=views[0].shape[0],
            )
        return objective
