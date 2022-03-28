import itertools
from typing import Optional

import numpy as np
import torch

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.architectures import Encoder
from cca_zoo.models import MCCA
from ._dcca_base import _DCCA_base
import matplotlib.pyplot as plt


class DCCA(_DCCA_base):
    """
    A class used to fit a DCCA model.

    :Citation:

    Andrew, Galen, et al. "Deep canonical correlation analysis." International conference on machine learning. PMLR, 2013.

    """

    def __init__(
        self,
        latent_dims: int,
        objective=objectives.MCCA,
        encoders=None,
        r: float = 0,
        eps: float = 1e-5,
        **kwargs,
    ):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        """
        super().__init__(latent_dims=latent_dims, **kwargs)
        self.encoders = torch.nn.ModuleList(encoders)
        self.objective = objective(latent_dims, r=r, eps=eps)

    def forward(self, *args, **kwargs):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return z

    def loss(self, *args):
        """
        Define the loss function for the model. This is used by the DeepWrapper class

        :param args:
        :return:
        """
        z = self(*args)
        return {"objective": self.objective.loss(*z)}

    def post_transform(self, z_list, train=False):
        if train:
            self.cca = MCCA(latent_dims=self.latent_dims)
            z_list = self.cca.fit_transform(z_list)
        else:
            z_list = self.cca.transform(z_list)
        return z_list

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        self.log(
            "train/corr", self.score(self.trainer.train_dataloader, train=True).sum()
        )

    def on_validation_epoch_end(self, unused: Optional = None) -> None:
        try:
            self.log("val/corr", self.score(self.trainer.val_dataloaders[0]).sum())
        except:
            # Should only be during sanity check
            self.log(
                "val/corr",
                self.score(self.trainer.val_dataloaders[0], train=True).sum(),
            )

    def pairwise_correlations(
        self,
        loader: torch.utils.data.DataLoader,
        train=False,
    ):
        """

        :param loader: a dataloader that matches the structure of that used for training
        :param train: whether to fit final linear transformation
        """
        transformed_views = self.transform(loader, train=train)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[: x.shape[1], y.shape[1] :]))
        all_corrs = np.array(all_corrs).reshape(
            (len(transformed_views), len(transformed_views), -1)
        )
        return all_corrs

    def score(
        self,
        loader: torch.utils.data.DataLoader,
        train=False,
    ):
        """

        :param loader: a dataloader that matches the structure of that used for training
        :param train: whether to fit final linear transformation
        :return: by default returns the average pairwise correlation in each dimension (for 2 views just the correlation)
        """
        pair_corrs = self.pairwise_correlations(loader, train=train)
        n_views = pair_corrs.shape[0]
        dim_corrs = (
            pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views
        ) / (n_views ** 2 - n_views)
        return dim_corrs

    def plot_latent_label(self, loader: torch.utils.data.DataLoader):
        fig, ax = plt.subplots(ncols=self.latent_dims)
        for j in range(self.latent_dims):
            ax[j].set_title(f"Dimension {j}")
            ax[j].set_xlabel("View 1")
            ax[j].set_ylabel("View 2")
        for i, batch in enumerate(loader):
            z = self(*batch["views"])
            zx, zy = z
            zx = zx.to("cpu").detach().numpy()
            zy = zy.to("cpu").detach().numpy()
            for j in range(self.latent_dims):
                im = ax[j].scatter(zx[:, j], zy[:, j], c=batch["label"].numpy())
