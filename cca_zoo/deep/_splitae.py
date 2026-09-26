"""Split autoencoder baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep._dccae import _reconstruction_loss


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
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.decoders = nn.ModuleList(decoders)

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The reconstruction loss of a batch.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective": loss}``.
        """
        views = batch["views"]
        joint = torch.cat(self(views), dim=1)
        return {
            "objective": _reconstruction_loss(
                views, [dec(joint) for dec in self.decoders]
            )
        }
