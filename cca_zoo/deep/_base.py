"""Lightning base class for the deep models."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn

from cca_zoo.linear._mcca import MCCA


class BaseDeep(pl.LightningModule):
    """Base class for deep multiview models, as Lightning modules.

    Subclasses implement :meth:`loss`; train with a
    :class:`lightning.pytorch.Trainer`.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        learning_rate: Adam learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.encoders = nn.ModuleList(encoders)

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode each view; one tensor of shape (batch_size, n_components) per view."""
        return [enc(v) for enc, v in zip(self.encoders, views)]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """The training loss of a batch.

        Args:
            representations: One encoded tensor per view.
            independent_representations: A second, independent batch's
                encodings, for losses that need one. Default is None.

        Returns:
            A dictionary whose ``"objective"`` entry is minimised.
        """
        raise NotImplementedError

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training loss of a batch.

        Args:
            batch: Dictionary with ``"views"`` and optionally
                ``"independent_views"``.
            batch_idx: Unused.

        Returns:
            The loss.
        """
        representations = self(batch["views"])
        ind_repr = (
            self(batch["independent_views"])
            if batch.get("independent_views") is not None
            else None
        )
        loss_dict = self.loss(representations, ind_repr)
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss_dict["objective"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation loss of a batch.

        Args:
            batch: Dictionary with ``"views"``.
            batch_idx: Unused.

        Returns:
            The loss.
        """
        representations = self(batch["views"])
        loss_dict = self.loss(representations)
        for k, v in loss_dict.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss_dict["objective"]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam with ``learning_rate``."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def transform(self, loader: torch.utils.data.DataLoader) -> list[np.ndarray]:
        """Encode every sample of a loader.

        Args:
            loader: DataLoader yielding batches with a ``"views"`` key.

        Returns:
            One array of shape (n_samples, n_components) per view.
        """
        self.eval()
        all_reprs: list[list[torch.Tensor]] = []
        for batch in loader:
            views_dev = [v.to(self.device) for v in batch["views"]]
            z = self(views_dev)
            all_reprs.append([zi.cpu() for zi in z])
        # Concatenate batches per view
        stacked = [
            torch.cat([b[i] for b in all_reprs], dim=0)
            for i in range(len(all_reprs[0]))
        ]
        return [t.numpy() for t in stacked]

    def score(self, loader: torch.utils.data.DataLoader) -> float:
        """Mean canonical correlation of linear CCA on the encodings.

        Args:
            loader: DataLoader yielding batches with a ``"views"`` key.

        Returns:
            The mean canonical correlation.
        """
        representations = self.transform(loader)
        return (
            MCCA(n_components=self.n_components)
            .fit(representations)
            .score(representations)
        )


def _inv_sqrtm(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute the inverse square root of a symmetric positive definite matrix.

    Args:
        A: Symmetric PD tensor of shape (n, n).
        eps: Regularisation added to eigenvalues for stability.

    Returns:
        Tensor of shape (n, n): $A^{-1/2}$.
    """
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    return V @ torch.diag(1.0 / torch.sqrt(L)) @ V.T
