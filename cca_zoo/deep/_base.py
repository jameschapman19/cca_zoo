"""Lightning base class for the deep models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike

from cca_zoo.linear._mcca import MCCA

Batch = dict[str, Any]


class BaseDeep(pl.LightningModule):
    """Base class for deep multiview models, as Lightning modules.

    Train with a :class:`lightning.pytorch.Trainer`. Calling the model returns
    each view's raw encoding. ``trainer.predict`` returns canonical variates:
    the encodings projected by a linear CCA fitted on the training encodings
    when training ends (:meth:`fit_cca`). Subclasses implement :meth:`loss`.

    Args:
        n_components: Latent dimension.
        encoders: One module per view, each mapping it to ``n_components``
            outputs.
        learning_rate: Adam learning rate. Default is 1e-3.

    Attributes:
        cca_weights: Linear CCA projection of each view's encoding, shape
            (n_views, n_components, n_components).
        cca_means: Training mean of each view's encoding, shape
            (n_views, n_components).
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["encoder", "encoders", "decoders", "objective"]
        )
        self.encoders = nn.ModuleList(encoders)
        self.n_components = n_components
        self.learning_rate = learning_rate
        n_views = len(encoders)
        self.register_buffer(
            "cca_weights", torch.eye(n_components).repeat(n_views, 1, 1)
        )
        self.register_buffer("cca_means", torch.zeros(n_views, n_components))
        self.register_buffer("cca_fitted", torch.tensor(False))
        self.cca_weights: torch.Tensor
        self.cca_means: torch.Tensor
        self.cca_fitted: torch.Tensor

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode each view; one tensor of shape (batch_size, n_components) per view."""
        return [self._checked(enc(v)) for enc, v in zip(self.encoders, views)]

    def _checked(self, z: torch.Tensor) -> torch.Tensor:
        """``z``, after checking its width is ``n_components``."""
        if z.shape[1] != self.n_components:
            raise ValueError(
                f"An encoder returned {z.shape[1]} outputs; n_components is "
                f"{self.n_components}."
            )
        return z

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The loss of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors, and optionally
                ``"independent_views"`` from an independent batch.

        Returns:
            A dictionary whose ``"objective"`` entry is minimised.
        """
        raise NotImplementedError

    def _logged_objective(self, batch: Batch, stage: str) -> torch.Tensor:
        """Compute the loss, log every term under ``stage`` and return the objective."""
        terms = self.loss(batch)
        batch_size = batch["views"][0].shape[0]
        for name, value in terms.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return terms["objective"]

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Training loss of a batch."""
        return self._logged_objective(batch, "train")

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Validation loss of a batch."""
        return self._logged_objective(batch, "val")

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Test loss of a batch."""
        return self._logged_objective(batch, "test")

    def predict_step(self, batch: Batch, batch_idx: int) -> list[torch.Tensor]:
        """Canonical variates of a batch, one tensor per view.

        Raises:
            RuntimeError: If the linear CCA has not been fitted.
        """
        if not self.cca_fitted:
            raise RuntimeError(
                "The linear CCA is not fitted; train with trainer.fit or call "
                "fit_cca(dataloader) first."
            )
        return [
            (z - mean) @ weights
            for z, mean, weights in zip(
                self(batch["views"]), self.cca_means, self.cca_weights
            )
        ]

    @torch.no_grad()
    def fit_cca(self, dataloader: Iterable[Batch]) -> None:
        """Fit the linear CCA that ``predict_step`` applies, on ``dataloader``.

        Called on the training data when training ends.

        Args:
            dataloader: Batches with a ``"views"`` key.
        """
        was_training = self.training
        self.eval()
        batches = [
            self([v.to(self.device) for v in batch["views"]]) for batch in dataloader
        ]
        self.train(was_training)
        encodings: list[ArrayLike] = [
            torch.cat(view_batches).cpu().numpy() for view_batches in zip(*batches)
        ]
        cca = MCCA(n_components=self.n_components).fit(encodings)
        self.cca_weights.copy_(torch.as_tensor(np.stack(cca.weights_)))
        self.cca_means.copy_(torch.as_tensor(np.stack(cca.means_)))
        self.cca_fitted.fill_(True)

    def on_train_end(self) -> None:
        """Fit the linear CCA on the training data."""
        dataloader = self.trainer.train_dataloader
        assert dataloader is not None
        self.fit_cca(dataloader)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam with ``learning_rate``."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def _require_two_views(encoders: list[nn.Module], name: str) -> None:
    """Raise unless there are exactly two encoders."""
    if len(encoders) != 2:
        raise ValueError(f"{name} is defined for two views, got {len(encoders)}.")
