"""BaseDeep — LightningModule base for all deep CCA models."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn

from cca_zoo.linear._mcca import MCCA


class BaseDeep(pl.LightningModule):
    """Base class for deep multiview CCA models using PyTorch Lightning.

    Subclasses override :meth:`loss` to implement the specific objective
    function.  Training is handled by a :class:`lightning.Trainer`.

    The sklearn-compatible interface (``fit``, ``transform``, ``score``) is
    provided for convenience, wrapping the Lightning training loop.

    Args:
        latent_dimensions: Dimensionality of the latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        lr: Learning rate for the Adam optimiser. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Small constant for numerical stability. Default is 1e-6.
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.latent_dimensions = latent_dimensions
        self.lr = lr
        self.max_epochs = max_epochs
        self.eps = eps
        self.encoders = nn.ModuleList(encoders)

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode all views into latent representations.

        Args:
            views: List of tensors, each (batch_size, n_features_i).

        Returns:
            List of tensors, each (batch_size, latent_dimensions).
        """
        return [enc(v) for enc, v in zip(self.encoders, views)]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the training objective.

        Args:
            representations: Encoded views from the current batch.
            independent_representations: Optional second set of encodings
                (e.g., for gradient correction in NOI).

        Returns:
            Dictionary with at least the key ``"objective"`` (to minimise).

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute the training loss for one mini-batch.

        Args:
            batch: Dictionary with key ``"views"`` (list of tensors) and
                optionally ``"independent_views"``.
            batch_idx: Batch index (unused).

        Returns:
            Scalar loss tensor.
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
        """Compute the validation loss for one mini-batch.

        Args:
            batch: Dictionary with ``"views"`` key.
            batch_idx: Batch index (unused).

        Returns:
            Scalar loss tensor.
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
        """Create the Adam optimiser.

        Returns:
            Adam optimiser with the configured learning rate.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def transform(self, loader: torch.utils.data.DataLoader) -> list[np.ndarray]:
        """Project all samples in a DataLoader into the latent space.

        Args:
            loader: DataLoader yielding batches with a ``"views"`` key.

        Returns:
            List of numpy arrays, each (n_samples, latent_dimensions).
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

    def score(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """Return average pairwise canonical correlations after linear CCA.

        Args:
            loader: DataLoader with a ``"views"`` key.

        Returns:
            Array of shape ``(latent_dimensions,)``.
        """
        representations = self.transform(loader)
        return (
            MCCA(latent_dimensions=self.latent_dimensions)
            .fit(representations)
            .score(representations)
        )


def _inv_sqrtm(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute the inverse square root of a symmetric positive definite matrix.

    Args:
        A: Symmetric PD tensor of shape (n, n).
        eps: Regularisation added to eigenvalues for stability.

    Returns:
        Tensor of shape (n, n): :math:`A^{-1/2}`.
    """
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    return V @ torch.diag(1.0 / torch.sqrt(L)) @ V.T
