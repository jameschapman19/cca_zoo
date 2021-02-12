"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""

from abc import abstractmethod
from typing import Iterable

from torch import nn
from torch import optim, matmul, mean, stack
from torch.linalg import norm
from cca_zoo.deep_models import BaseEncoder, Encoder
from cca_zoo.objectives import compute_matrix_power, CCA


class DCCA_base(nn.Module):
    def __init__(self, latent_dims: int, post_transform=False):
        super(DCCA_base, self).__init__()
        self.latent_dims = latent_dims
        self.post_transform = post_transform
        self.schedulers=[None]

    @abstractmethod
    def update_weights(self, *args):
        """
        A complete update of the weights used every batch
        :param args: batches for each view separated by commas
        :return:
        """
        pass

    @abstractmethod
    def forward(self, *args):
        """
        :param args: batches for each view separated by commas
        :return: views encoded to latent dimensions
        """
        pass


class DCCA(DCCA_base, nn.Module):
    def __init__(self, latent_dims: int, objective=CCA,
                 encoders: Iterable[BaseEncoder] = (Encoder, Encoder),
                 learning_rate=1e-3, als=False, rho: float = 0.2, eps: float = 1e-9, post_transform=True,
                 shared_target=False, schedulers=None, optimizers=None):
        """
        :param latent_dims:
        :param objective:
        :param encoders:
        :param learning_rate:
        :param als:
        :param rho:
        :param eps:
        :param post_transform:
        :param shared_target:
        :param schedulers: list of pytorch.optim optimizer classes
        :param optimizers: list of pytorch.optim scheduler classes
        """
        super().__init__(latent_dims, post_transform=post_transform)
        self.latent_dims = latent_dims
        self.encoders = nn.ModuleList(encoders)
        self.objective = objective(latent_dims)
        if optimizers is None:
            self.optimizers = [optim.Adam(list(encoder.parameters()), lr=learning_rate) for encoder in self.encoders]
        else:
            self.optimizers = list(optimizers)
        self.schedulers = list(schedulers)
        self.schedulers=filter(None, self.schedulers)
        self.covs = None
        self.eps = eps
        self.rho = rho
        self.als = als
        self.shared_target = shared_target
        if self.als:
            assert (0 <= self.rho <= 1), "rho should be between 0 and 1"
        elif self.shared_target:
            assert (0 <= self.rho <= 1), "rho should be between 0 and 1"

    def update_weights(self, *args):
        if self.als:
            loss = self.update_weights_als(*args)
        else:
            loss = self.update_weights_tn(*args)
        return loss

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return tuple(z)

    def loss(self, *args):
        z = self(*args)
        return self.objective.loss(*z)

    def update_weights_tn(self, *args):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        loss = self.loss(*args)
        loss.backward()
        [optimizer.step() for optimizer in self.optimizers]
        return loss

    def update_weights_als(self, *args):
        losses, obj = self.als_loss(*args)
        self.optimizers[0].zero_grad()
        losses[0].backward()
        self.optimizers[0].step()
        self.optimizers[1].zero_grad()
        losses[1].backward()
        self.optimizers[1].step()
        return obj

    def als_loss(self, *args):
        z = self(*args)
        self.update_covariances(*z)
        covariance_inv = [compute_matrix_power(cov, -0.5, self.eps) for cov in self.covs]
        preds = [matmul(z, covariance_inv[i]).detach() for i, z in enumerate(z)]
        losses = [mean(norm(z_i - preds[-i], dim=0)) for i, z_i in enumerate(z, start=1)]
        obj = self.objective.loss(*z)
        return losses, obj

    def update_covariances(self, *args):
        b = args[0].shape[0]
        batch_covs = [z_i.T @ z_i for i, z_i in enumerate(args)]
        if self.covs is not None:
            self.covs = [(self.rho * self.covs[i]).detach() + (1 - self.rho) * batch_cov for i, batch_cov
                         in
                         enumerate(batch_covs)]
        else:
            self.covs = batch_covs
