import numpy as np
import torch

from cca_zoo.linear._gradient._base import BaseGradientModel
from cca_zoo.linear._pls import PLSMixin


class CCAEY(BaseGradientModel):
    def _more_tags(self):
        return {"multiview": True, "stochastic": True}

    def training_step(self, batch, batch_idx):
        if self.batch_size is None:
            batch = self.batch
        else:
            self._initialize_queue()
        loss = self._compute_loss(batch)
        # Logging the loss components
        for k, v in loss.items():
            self.log(k, v, prog_bar=True)
        return loss

    def _initialize_queue(self):
        if not hasattr(self, "batch_queue"):
            self.batch_queue = []
            self.val_batch_queue = []

    def _get_random_batch(self) -> dict:
        return self.batch_queue[np.random.randint(0, len(self.batch_queue))]

    def _update_queue(self, batch):
        self.batch_queue.append(batch)
        self.batch_queue.pop(0)

    def _compute_loss(self, batch) -> dict:
        if self.batch_size is None:
            loss = self.loss(batch["views"])
        else:
            if len(self.batch_queue) < 5:
                self.batch_queue.append(batch)
                return {
                    "loss": torch.tensor(0, requires_grad=True, dtype=torch.float32)
                }
            else:
                random_batch = self._get_random_batch()
                loss = self.loss(batch["views"], random_batch["views"])
            self._update_queue(batch)
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

    def loss(self, views, views2=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)

        if views2 is None:
            # Computing rewards and penalties using A and B only
            rewards = torch.trace(2 * A)
            penalties = torch.trace(B @ B)

        else:
            # Encoding another set of views with the forward method
            z2 = self(views2)
            # Getting A' and B' matrices from z2
            A_, B_ = self.get_AB(z2)
            # Computing rewards and penalties using A and B'
            rewards = torch.trace(2 * A)
            penalties = torch.trace(B @ B_)

        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }


class PLSEY(CCAEY, PLSMixin):
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
                    B += self.torch_weights[i].T @ self.torch_weights[i]
                else:
                    A += self._cross_covariance(zi, zj, latent_dims)
        return A / len(z), B / len(z)
