"""Lightning base class for the deep models."""

from __future__ import annotations

import inspect
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn

Batch = dict[str, Any]


class BaseDeep(pl.LightningModule):
    """Base class for deep multiview models, as Lightning modules.

    Train with a :class:`lightning.pytorch.Trainer`. Calling the model, or
    ``trainer.predict``, returns each view's encoding; for canonical variates,
    fit :class:`~cca_zoo.linear.CCA` or :class:`~cca_zoo.linear.MCCA` to the
    training encodings. Subclasses implement :meth:`loss`.

    The covariance-based losses are estimated from each process's batch, so
    under multi-GPU (DDP) training they see the per-GPU batch, not the global
    one.

    Args:
        n_components: Number of latent dimensions.
        encoders: One module per view, each mapping it to ``n_components``
            outputs.
        learning_rate: Adam learning rate. Default is 1e-3.

    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        # Modules are not hyperparameters: they are saved in the state dict and
        # passed back to load_from_checkpoint.
        self.save_hyperparameters(
            ignore=[
                name
                for cls in type(self).__mro__
                if "__init__" in vars(cls)
                for name, param in inspect.signature(cls.__init__).parameters.items()
                if "Module" in str(param.annotation)
            ]
        )
        self.encoders = nn.ModuleList(encoders)
        self.n_components = n_components
        self.learning_rate = learning_rate

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
        """Encodings of a batch, one tensor per view."""
        return self.forward(batch["views"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam with ``learning_rate``."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
