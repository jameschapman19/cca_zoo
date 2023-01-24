from abc import abstractmethod

import pytorch_lightning as pl
import torch
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

    def transform(
        self,
        loader: torch.utils.data.DataLoader,
    ):
        """
        :param loader: a dataloader that matches the structure of that used for training
        :return: transformed views
        """
        with torch.no_grad():
            z = []
            for batch_idx, batch in enumerate(loader):
                views = [view.to(self.device) for view in batch["views"]]
                z_ = self(views)
                z.append(z_)
        z = [torch.vstack(i).cpu().numpy() for i in zip(*z)]
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

    @staticmethod
    def detach_all(z):
        [z_.detach() for z_ in z]
        return z
