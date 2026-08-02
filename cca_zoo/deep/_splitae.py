"""SplitAE — Split Autoencoder baseline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import BaseDeep


class SplitAE(BaseDeep):
    r"""Split Autoencoder baseline for multiview learning.

    All views are encoded individually into a shared latent space.
    The concatenated representations are used to reconstruct each
    view via dedicated decoders:

    $$
    \mathcal{L} = \sum_i \operatorname{MSE}\!\bigl(
        x_i,\ \text{decoder}_i(z_1 \,\|\, \cdots \,\|\, z_V)
    \bigr)
    $$

    where $\|$ denotes concatenation. This model serves as a
    reconstruction-based baseline that does not explicitly maximise
    correlation between views — "split autoencoder" architectures of this
    kind are a common baseline in the deep multiview representation
    learning literature (e.g. compared against by Wang, W., et al. in
    "On deep multi-view representation learning." ICML 2015), rather than
    a single-paper reproduction.

    Args:
        latent_dimensions: Dimensionality of each encoder's output.
            Decoders receive the concatenation of all encoder outputs,
            so their input size is n_views * latent_dimensions.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        decoders: List of :class:`torch.nn.Module` objects.  Each
            decoder's input dimension should be
            n_views * latent_dimensions.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Unused; present for API consistency. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> # Decoders receive 2 * 4 = 8 dimensional input
        >>> dec1 = nn.Linear(8, 10)
        >>> dec2 = nn.Linear(8, 8)
        >>> model = SplitAE(
        ...     latent_dimensions=4,
        ...     encoders=[enc1, enc2],
        ...     decoders=[dec1, dec2],
        ... )
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        decoders: list[nn.Module],
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.decoders = nn.ModuleList(decoders)

    def _decode(self, representations: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode using the concatenation of all latent representations.

        Args:
            representations: List of latent tensors, each of shape
                (batch_size, latent_dimensions).

        Returns:
            List of reconstructed tensors, one per decoder.
        """
        z_cat = torch.cat(representations, dim=-1)
        return [dec(z_cat) for dec in self.decoders]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a zero loss placeholder.

        Reconstruction requires the original views; use the full
        training step for proper loss computation.

        Args:
            representations: Unused.
            independent_representations: Unused.

        Returns:
            Dictionary with ``"objective"`` set to 0.
        """
        return {"objective": torch.tensor(0.0)}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step computing the reconstruction loss.

        Args:
            batch: Dictionary with key ``"views"`` (list of tensors).
            batch_idx: Batch index (unused).

        Returns:
            Scalar loss tensor.
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
