from abc import abstractmethod
from typing import Iterable

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


class _BaseDeep(pl.LightningModule):
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
    def forward(self, views, *args, **kwargs):
        """
        We use the forward model to define the transformation of views to the latent space
        :param views: batches for each view separated by commas
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, views, *args, **kwargs):
        """
        Loss for model
        :param views:
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            self.log("train/" + k, v, prog_bar=True)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            self.log("val/" + k, v)
        return loss["objective"]

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch["views"])
        for k, v in loss.items():
            self.log("test/" + k, v)
        return loss["objective"]

    def post_transform(self, z, train=False):
        """
        Some models require a final linear CCA after model training.
        :param z: a list of all of the latent space embeddings for each view
        :param train: if the train flag is True this fits a new post transformation
        """
        return z

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
                z_ = detach_all(self(views))
                if batch_idx == 0:
                    z = z_
                else:
                    z = collate_all(z, z_)
        z = self.post_transform(z, train=train)
        return z

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

    def configure_callbacks(self):
        pass


class _GenerativeMixin:
    def recon_loss(self, x, recon, loss="mse", reduction="mean", **kwargs):
        if loss == "mse":
            return self.mse_loss(x, recon, reduction=reduction)
        elif loss == "bce":
            return self.mse_loss(x, recon, reduction=reduction)
        elif loss == "nll":
            return self.mse_loss(x, recon, reduction=reduction)

    def mse_loss(self, x, recon, reduction="mean"):
        return F.mse_loss(recon, x, reduction=reduction)

    def bce_loss(self, x, recon, reduction="mean"):
        return F.binary_cross_entropy(recon, x, reduction=reduction)

    def nll_loss(self, x, recon, reduction="mean"):
        return F.nll_loss(recon, x, reduction=reduction)

    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @abstractmethod
    def _decode(self, z, **kwargs):
        raise NotImplementedError

    def recon(self,
            loader: torch.utils.data.DataLoader, **kwargs):
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views = [view.to(self.device) for view in batch["views"]]
                x_ = detach_all(self._decode(self(views, **kwargs)))
                if batch_idx == 0:
                    x = x_
                else:
                    x = collate_all(x, x_)
        return x


def detach_all(z):
    if isinstance(z, dict):
        for k, v in z.items():
            detach_all(v)
    elif isinstance(z, list):
        z = [z_.detach().cpu().numpy() for z_ in z]
    else:
        z = z.detach().cpu().numpy()
    return z


def collate_all(z, z_):
    if isinstance(z, dict):
        for k, v in z_.items():
            z[k] = collate_all(z[k], v)
    elif isinstance(z, list):
        z = [np.append(z[i], z_i, axis=0) for i, z_i in enumerate(z_)]
    else:
        z = np.append(z, z_, axis=0)
    return z