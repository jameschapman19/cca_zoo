from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch

from cca_zoo._base import _BaseModel
from cca_zoo.linear._mcca import MCCA


class BaseDeep(pl.LightningModule, _BaseModel):
    """A base class for deep learning linear using PyTorch Lightning."""

    def __init__(
        self,
        latent_dimensions: int,
        encoders=None,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        lr: float = 1e-2,
        extra_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_epochs: int = 1000,
        eps=1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        if extra_optimizer_kwargs is None:
            extra_optimizer_kwargs = {}
        self.latent_dimensions = latent_dimensions
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.extra_optimizer_kwargs = extra_optimizer_kwargs
        self.max_epochs = max_epochs
        self.eps = eps
        if encoders is None:
            raise ValueError(
                "Encoders must be a list of torch.nn.Module with length equal to the number of representations."
            )
        self.encoders = torch.nn.ModuleList(encoders)

    def forward(self, views, **kwargs):
        if not hasattr(self, "n_views_"):
            self.n_views_ = len(views)
        # Use list comprehension to encode each view
        representations = [encoder(view) for encoder, view in zip(self.encoders, views)]
        return representations

    def minibatch_loss(self, batch, **kwargs):
        # Encoding the representations with the forward method
        representations = self(batch["views"])
        if batch.get("independent_views") is None:
            independent_representations = None
        else:
            independent_representations = self(batch["independent_views"])
        return self.loss(representations, independent_representations)

    def pairwise_correlations(self, loader: torch.utils.data.DataLoader):
        # Call the parent class method
        return super().pairwise_correlations(loader)

    def correlation_captured(self, representations):
        # Remove mean from each view
        representations = [
            representation - representation.mean(0)
            for representation in representations
        ]
        return (
            MCCA(latent_dimensions=self.latent_dimensions)
            .fit(representations)
            .score(representations)
            .sum()
        )

    def score(self, loader: torch.utils.data.DataLoader, **kwargs):
        representations = self.transform(loader)
        corr = self.correlation_captured(representations)
        return corr

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs one step of training on a batch of representations."""
        loss = self.minibatch_loss(batch)
        for k, v in loss.items():
            # Use f-string instead of concatenation
            self.log(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss["objective"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs one step of validation on a batch of representations."""
        loss = self.minibatch_loss(batch)
        for k, v in loss.items():
            # Use f-string instead of concatenation
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss["objective"]

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs one step of testing on a batch of representations."""
        loss = self.minibatch_loss(batch)
        for k, v in loss.items():
            # Use f-string instead of concatenation
            self.log(
                f"test/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss["objective"]

    @torch.no_grad()
    def transform(
        self,
        loader: torch.utils.data.DataLoader,
    ):
        self.eval()  # Ensure the model is in evaluation mode
        representations = []

        for batch in loader:
            views_device = [view.to(self.device) for view in batch["views"]]
            z = self(views_device)
            representations.append([z_.cpu().detach() for z_ in z])

        # Stack all latent vectors along dimension 0 (batches)
        representations = [torch.vstack(z_) for z_ in zip(*representations)]

        return [representation.numpy() for representation in representations]

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]],
    ]:
        """Configures the optimizer and the learning rate scheduler."""
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        elif self.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

        # create optimizer
        optimizer = optimizer(
            self.parameters(),
            lr=self.lr,
            **self.extra_optimizer_kwargs,
        )
        return optimizer

    def configure_callbacks(self) -> None:
        """Configures the callbacks for the model."""
        pass
