from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

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
        lr: float = 1e-3,
        weight_decay: float = 0,
        extra_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_epochs: int = 1000,
        min_lr: float = 1e-9,
        eps=1e-6,
        lr_decay_steps: Optional[List[int]] = None,
        correlation: bool = True,
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
        self.weight_decay = weight_decay
        self.extra_optimizer_kwargs = extra_optimizer_kwargs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.lr_decay_steps = lr_decay_steps
        self.correlation = correlation
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
        z = [encoder(view) for encoder, view in zip(self.encoders, views)]
        return z

    def loss(self, batch, **kwargs):
        representations = self(batch["views"])
        return {"objective": self.objective(representations)}

    def pairwise_correlations(self, loader: torch.utils.data.DataLoader):
        # Call the parent class method
        return super().pairwise_correlations(loader)

    def correlation_captured(self, z):
        # Remove mean from each view
        z = [zi - zi.mean(0) for zi in z]
        return MCCA(latent_dimensions=self.latent_dimensions).fit(z).score(z).sum()

    def score(self, loader: torch.utils.data.DataLoader, **kwargs):
        z = self.transform(loader)
        corr = self.correlation_captured(z)
        return corr

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs one step of training on a batch of representations."""
        loss = self.loss(batch)
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
        loss = self.loss(batch)
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
        loss = self.loss(batch)
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
    def get_representations(
        self,
        loader: torch.utils.data.DataLoader,
    ):
        self.eval()  # Ensure the model is in evaluation mode
        all_z = []

        for batch in loader:
            views_device = [view.to(self.device) for view in batch["views"]]
            z = self(views_device)
            all_z.append([z_.cpu() for z_ in z])

        # Stack all latent vectors along dimension 0 (batches)
        stacked_z = [torch.vstack(z_) for z_ in zip(*all_z)]

        return stacked_z

    @torch.no_grad()
    def transform(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> List[np.ndarray]:
        """Returns the latent representations for each view in the loader."""
        return [z.numpy() for z in self.get_representations(loader)]

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
        if self.scheduler is None:
            return optimizer
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, self.max_epochs, eta_min=self.min_lr
            )
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")
        return [optimizer], [scheduler]

    def configure_callbacks(self) -> None:
        """Configures the callbacks for the model."""
        pass

    @staticmethod
    def detach_all(z: List[torch.Tensor]) -> List[torch.Tensor]:
        """Detaches all tensors in a list from the computation graph."""
        # Use list comprehension instead of for loop
        return [z_.detach() for z_ in z]
