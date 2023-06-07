from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


class BaseDeep(pl.LightningModule):
    """A base class for deep learning models using PyTorch Lightning."""

    def __init__(
        self,
        latent_dims: int,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        lr: float = 1e-2,
        weight_decay: float = 0,
        extra_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_epochs: int = 1000,
        min_lr: float = 1e-9,
        lr_decay_steps: Optional[List[int]] = None,
        correlation: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        if extra_optimizer_kwargs is None:
            extra_optimizer_kwargs = {}
        self.latent_dims = latent_dims
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_kwargs = extra_optimizer_kwargs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.lr_decay_steps = lr_decay_steps
        self.correlation = correlation

    @abstractmethod
    def forward(self, views: List[torch.Tensor], *args, **kwargs) -> List[torch.Tensor]:
        """Returns the latent representations for each view."""
        raise NotImplementedError

    @abstractmethod
    def loss(
        self, views: List[torch.Tensor], *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Returns the loss components for each view."""
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs one step of training on a batch of views."""
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            # Use f-string instead of concatenation
            self.log(f"train/{k}", v, prog_bar=True)
        return loss["objective"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs one step of validation on a batch of views."""
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            # Use f-string instead of concatenation
            self.log(f"val/{k}", v)
        return loss["objective"]

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs one step of testing on a batch of views."""
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            # Use f-string instead of concatenation
            self.log(f"test/{k}", v)
        return loss["objective"]

    def transform(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> List[np.ndarray]:
        """Returns the latent representations for each view in the loader."""
        with torch.no_grad():
            # Use list comprehension instead of for loop
            z = [
                self([view.to(self.device) for view in batch["views"]])
                for batch in loader
            ]
        # Use list comprehension instead of for loop
        z = [torch.vstack(i).cpu().numpy() for i in zip(*z)]
        return z

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

    def _more_tags(self) -> Dict[str, bool]:
        """Returns additional tags for the model."""
        return {"multiview": True}
