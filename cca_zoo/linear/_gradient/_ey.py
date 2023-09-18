from typing import Iterable
import numpy as np
import torch
from torch.utils import data

from cca_zoo.data.deep import NumpyDataset
from cca_zoo.linear._gradient._base import BaseGradientModel, FullBatchDataset
from cca_zoo.linear._pls import PLSMixin


class CCA_EY(BaseGradientModel):
    def _more_tags(self):
        return {"multiview": True, "stochastic": True}

    def training_step(self, batch, batch_idx):
        if self.batch_size is None:
            batch = self.batch
        loss = self.loss(batch["views"], batch.get("independent_views", None))
        # Logging the loss components
        for k, v in loss.items():
            self.log(k, v, prog_bar=True)
        return loss

    def get_AB(self, z):
        latent_dims = z[0].shape[1]
        A = torch.zeros(
            latent_dims, latent_dims, device=z[0].device
        )  # initialize the cross-covariance matrix
        B = torch.zeros(
            latent_dims, latent_dims, device=z[0].device
        )  # initialize the auto-covariance matrix
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    B += self._cross_covariance(zi, zj, latent_dims)
                else:
                    A += self._cross_covariance(zi, zj, latent_dims)
        return A / len(z), B / len(z)

    @staticmethod
    def _cross_covariance(zi, zj, latent_dims) -> torch.Tensor:
        return torch.cov(torch.hstack((zi, zj)).T)[latent_dims:, :latent_dims]

    def loss(self, views, independent_views=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)
        if independent_views is None:
            independent_B = B
        else:
            # Encoding another set of views with the forward method
            independent_z = self(independent_views)
            # Getting A' and B' matrices from independent_z
            independent_A, independent_B = self.get_AB(independent_z)
        # Computing rewards and penalties using A and B'
        rewards = torch.trace(2 * A)
        penalties = torch.trace(B @ independent_B)

        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    def get_dataset(self, views: Iterable[np.ndarray]):
        dataset = (
            DoubleNumpyDataset(views) if self.batch_size else FullBatchDataset(views)
        )
        if self.val_split:
            train_size = int((1 - self.val_split) * len(dataset))
            val_size = len(dataset) - train_size
            dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
        else:
            val_dataset = None
        return dataset, val_dataset


class DoubleNumpyDataset(NumpyDataset):
    random_state = np.random.RandomState(0)

    def __getitem__(self, index):
        views = [view[index] for view in self.views]
        independent_index = self.random_state.randint(0, len(self))
        independent_views = [view[independent_index] for view in self.views]
        return {"views": views, "independent_views": independent_views}


class PLS_EY(CCA_EY, PLSMixin):
    def get_AB(self, z):
        latent_dims = z[0].shape[1]
        A = torch.zeros(
            latent_dims, latent_dims, device=z[0].device
        )  # initialize the cross-covariance matrix
        B = torch.zeros(
            latent_dims, latent_dims, device=z[0].device
        )  # initialize the auto-covariance matrix
        n = z[0].shape[0]
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    B += self.torch_weights[i].T @ self.torch_weights[i] / n
                else:
                    A += self._cross_covariance(zi, zj, latent_dims)
        return A / len(z), B / len(z)
