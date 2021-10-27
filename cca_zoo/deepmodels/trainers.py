import itertools
import sys
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


class CCALightning(LightningModule):
    def __init__(
            self,
            model,
            optimizer='Adam',
            learning_rate=1e-3,
            weight_decay=0.1,
            lr_scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def loss(self, *args, **kwargs):
        return self.model.loss(*args, **kwargs)

    # Configuration.  Add more for learning schedulers, etc.?
    def configure_optimizers(self):
        if isinstance(self.hparams.optimizer, torch.optim.Optimizer):
            optimizer = self.hparams.optimizer
        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "SGD":
            # Left out the momentum options for now
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "LBFGS":
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                # or can have self.hparams.learning_rate with warning if too low.
                lr=1,
                tolerance_grad=1e-5,  # can add to parameters if useful.
                tolerance_change=1e-9,  # can add to parameters if useful.
            )
        else:
            print("Invalid optimizer.  See --help")
            sys.exit()

        if self.hparams.lr_scheduler == None:
            return optimizer
        elif isinstance(self.hparams.lr_scheduler, torch.optim.lr_scheduler):
            scheduler = self.hparams.lr_scheduler
        elif self.hparams.lr_scheduler == "StepLR":
            step_size = self.hparams.StepLR_step_size
            gamma = self.hparams.StepLR_gamma
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size, gamma)
        elif self.hparams.lr_scheduler == "ReduceLROnPlateau":
            factor = self.hparams.lr_factor
            patience = self.hparams.lr_patience
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.hparams.LRScheduler_metric,
            }
        elif self.hparams.lr_scheduler == "OneCycleLR":
            max_lr = self.hparams.OneCycleLR_max_lr
            epochs = self.hparams.OneCycleLR_epochs
            steps_per_epoch = self.hparams.train_trajectories * (
                    self.hparams.T + 1
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
            )
        else:
            print("Invalid scheduler configuration.  See --help")
            raise
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
        score = self.score(
            self.train_dataloader(), train=True
        ).sum()
        self.log('train corr', score)

    def on_validation_epoch_end(self, unused: Optional = None) -> None:
        score = self.score(
            self.val_dataloader(), train=True
        ).sum()
        self.log('val corr', score)

    def correlations(
            self,
            loader: torch.utils.data.DataLoader,
            train: bool = False,
    ):
        """


        :return: numpy array containing correlations between each pair of views for each dimension (#views*#views*#latent_dimensions)
        """
        transformed_views = self.transform(
            loader, train=train
        )
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
        z_list = self.model.post_transform(*z_list, train=train)
        return z_list

    def score(
            self,
            loader: torch.utils.data.DataLoader,
            train: bool = False,
    ):
        # by default return the average pairwise correlation in each dimension (for 2 views just the correlation)
        pair_corrs = self.correlations(
            loader, train=train
        )
        # n views
        n_views = pair_corrs.shape[0]
        # sum all the pairwise correlations for each dimension. Subtract the self correlations. Divide by the number of views. Gives average correlation
        dim_corrs = (pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views
                     ) / (n_views ** 2 - n_views)
        return dim_corrs

    def predict_view(
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
