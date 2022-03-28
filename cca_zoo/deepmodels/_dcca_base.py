from abc import abstractmethod
from typing import Iterable

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


class _DCCA_base(pl.LightningModule):
    def __init__(
        self,
        latent_dims: int,
        optimizer="adam",
        scheduler=None,
        lr=1e-3,
        weight_decay=0,
        extra_optimizer_kwargs=None,
        max_epochs=1000,
        min_lr=1e-9,
        lr_decay_steps=None,
        correlation=True,
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
    def forward(self, *args, **kwargs):
        """
        We use the forward model to define the transformation of views to the latent space

        :param args: batches for each view separated by commas
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, *args, **kwargs):
        """
        Required when using the LightningTrainer
        """
        raise NotImplementedError

    def post_transform(self, z_list, train=False) -> Iterable[np.ndarray]:
        """
        Some models require a final linear CCA after model training.

        :param z_list: a list of all of the latent space embeddings for each view
        :param train: if the train flag is True this fits a new post transformation
        """
        return z_list

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch["views"])
        for k, v in loss.items():
            self.log("train/" + k, v, prog_bar=True)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch["views"])
        for k, v in loss.items():
            self.log("val/" + k, v)
        return loss["objective"]

    def test_step(self, batch, batch_idx):
        loss = self.loss(*batch["views"])
        for k, v in loss.items():
            self.log("test/" + k, v)
        return loss["objective"]

    def transform(
        self,
        loader: torch.utils.data.DataLoader,
        train=False,
    ):
        """

        :param loader: a dataloader that matches the structure of that used for training
        :param train: whether to fit final linear transformation
        :return: transformed views
        """
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views = [view.to(self.device) for view in batch["views"]]
                z = self(*views)
                if isinstance(z[0], dict):
                    z = z[0]["shared"]
                if batch_idx == 0:
                    z_list = [z_i.detach().cpu().numpy() for i, z_i in enumerate(z)]
                else:
                    z_list = [
                        np.append(z_list[i], z_i.detach().cpu().numpy(), axis=0)
                        for i, z_i in enumerate(z)
                    ]
        z_list = self.post_transform(z_list, train=train)
        return z_list

    def configure_optimizers(self):
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.
        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """
        # select optimizer
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

    def on_train_end(self) -> None:
        pass
