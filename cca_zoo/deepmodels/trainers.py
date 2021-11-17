import itertools
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from cca_zoo.deepmodels import _DCCA_base


class CCALightning(LightningModule):
    def __init__(
            self,
            model: _DCCA_base,
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler = None,
    ):
        """

        :param model: a model instance from deepmodels
        :param optimizer: a pytorch optimizer with parameters from model
        :param lr_scheduler: a pytorch scheduler
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def loss(self, *args, **kwargs):
        return self.model.loss(*args, **kwargs)

    def configure_optimizers(self):
        if isinstance(self.hparams.optimizer, torch.optim.Optimizer):
            optimizer = self.hparams.optimizer
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=1e-3,
            )

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            scheduler = self.hparams.lr_scheduler
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data, label = batch
        loss = self.model.loss(*data)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        loss = self.model.loss(*data)
        return loss

    def test_step(self, batch, batch_idx):
        data, label = batch
        loss = self.model.loss(*data)
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        score = self.score(self.trainer.train_dataloader, train=True).sum()
        self.log("train corr", score)

    def on_validation_epoch_end(self, unused: Optional = None) -> None:
        score = self.score(self.trainer.val_dataloaders[0], train=True).sum()
        self.log("val corr", score)

    def correlations(
            self,
            loader: torch.utils.data.DataLoader,
            train: bool = False,
    ):
        """

        :param loader: a dataloader that matches the structure of that used for training
        :param train: if True and the model requires a final linear CCA this solves and stores the linear CCA
        :return: numpy array containing correlations between each pair of views for each dimension (#views*#views*#latent_dimensions)
        """
        transformed_views = self.transform(loader, train=train)
        if len(transformed_views) < 2:
            return None
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[: x.shape[1], y.shape[1]:]))
        all_corrs = np.array(all_corrs).reshape(
            (len(transformed_views), len(transformed_views), -1)
        )
        return all_corrs

    def transform(
            self,
            loader: torch.utils.data.DataLoader,
            train: bool = False,
    ):
        """

        :param loader: a dataloader that matches the structure of that used for training
        :param train: if True and the model requires a final linear CCA this solves and stores the linear CCA
        :return: transformed views
        """
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                data = [d.to(self.device) for d in list(data)]
                z = self.model(*data)
                if batch_idx == 0:
                    z_list = [z_i.detach().cpu().numpy() for i, z_i in enumerate(z)]
                else:
                    z_list = [
                        np.append(z_list[i], z_i.detach().cpu().numpy(), axis=0)
                        for i, z_i in enumerate(z)
                    ]
        z_list = self.model.post_transform(z_list, train=train)
        return z_list

    def score(
            self,
            loader: torch.utils.data.DataLoader,
            train: bool = False,
    ):
        """

        :param loader: a dataloader that matches the structure of that used for training
        :param train: if True and the model requires a final linear CCA this solves and stores the linear CCA
        :return: by default returns the average pairwise correlation in each dimension (for 2 views just the correlation)
        """
        pair_corrs = self.correlations(loader, train=train)
        if pair_corrs is None:
            return np.zeros(1)
        # n views
        n_views = pair_corrs.shape[0]
        # sum all the pairwise correlations for each dimension. Subtract the self correlations. Divide by the number of views. Gives average correlation
        dim_corrs = (
                            pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views
                    ) / (n_views ** 2 - n_views)
        return dim_corrs

    def recon(
            self,
            loader: torch.utils.data.DataLoader,
    ):
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                data = [d.to(self.device) for d in list(data)]
                x = self.model.recon(*data)
                if batch_idx == 0:
                    x_list = [x_i.detach().cpu().numpy() for i, x_i in enumerate(x)]
                else:
                    x_list = [
                        np.append(x_list[i], x_i.detach().cpu().numpy(), axis=0)
                        for i, x_i in enumerate(x)
                    ]
        return x_list
