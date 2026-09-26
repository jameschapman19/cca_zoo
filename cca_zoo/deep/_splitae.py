"""Split autoencoder baseline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import BaseDeep


class SplitAE(BaseDeep):
    r"""Split autoencoder: every view reconstructed from all views' encodings.

    $$
    \mathcal{L} = \sum_i \operatorname{MSE}(x_i, \text{dec}_i(z_1 \| \cdots \| z_M)).
    $$

    A reconstruction baseline that does not maximise correlation.

    Args:
        n_components: Dimension of each encoding.
        encoders: One module per view.
        decoders: One module per view, taking ``n_views * n_components``
            inputs.
        learning_rate: Adam learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.

    References:
        Wang, W., Arora, R., Livescu, K., & Bilmes, J. (2015). On deep
        multi-view representation learning. ICML.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import SplitAE
        >>> model = SplitAE(
        ...     n_components=4,
        ...     encoders=[nn.Linear(10, 4), nn.Linear(8, 4)],
        ...     decoders=[nn.Linear(8, 10), nn.Linear(8, 8)],
        ... )
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        decoders: list[nn.Module],
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
        )
        self.decoders = nn.ModuleList(decoders)

    def _decode(self, representations: list[torch.Tensor]) -> list[torch.Tensor]:
        """Reconstruct every view from the concatenated encodings."""
        z_cat = torch.cat(representations, dim=-1)
        return [dec(z_cat) for dec in self.decoders]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Zero objective; the loss needs the inputs, so it is computed in training.

        Args:
            representations: Unused.
            independent_representations: Unused.

        Returns:
            ``{"objective": 0}``.
        """
        return {"objective": torch.tensor(0.0)}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Reconstruction loss of a batch.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.
            batch_idx: Unused.

        Returns:
            The loss.
        """
        views = batch["views"]
        representations = self(views)
        reconstructions = self._decode(representations)

        recon_loss = torch.stack(
            [F.mse_loss(x, r) for x, r in zip(views, reconstructions)]
        ).sum()
        self.log(
            "train/objective",
            recon_loss,
            on_step=False,
            on_epoch=True,
            batch_size=views[0].shape[0],
        )
        return recon_loss
