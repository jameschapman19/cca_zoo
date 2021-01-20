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
        self.post_transform=post_transform
    @abstractmethod
    def update_weights(self, *args):
        pass

    @abstractmethod
    def forward(self, *args):
        pass


class DCCA(DCCA_base):
    def __init__(self, latent_dims: int, objective=CCA,
                 encoders: Iterable[BaseEncoder] = (Encoder, Encoder),
                 learning_rate=1e-3, als=False, rho: float = 0.2, eps: float = 1e-9,post_transform=True):
        super().__init__(latent_dims,post_transform=post_transform)
        self.latent_dims = latent_dims
        self.encoders = nn.ModuleList(encoders)
        self.objective = objective(latent_dims)
        self.optimizers = [optim.Adam(list(encoder.parameters()), lr=learning_rate) for encoder in self.encoders]
        self.covs = None
        self.eps = eps
        self.rho = rho
        if als:
            assert (0 <= self.rho <= 1), "rho should be between 0 and 1"
            self.update_weights = self.update_weights_als
        else:
            self.update_weights = self.update_weights_tn

    @abstractmethod
    def update_weights(self, *args):
        if self.als:
            loss = self.update_weights_als(*args)
        else:
            loss = self.update_weights_tn(*args)
        return loss

    @abstractmethod
    def forward(self, *args):
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return tuple(z)

    def update_weights_tn(self, *args):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        loss = self.loss(*args)
        loss.backward()
        [optimizer.step() for optimizer in self.optimizers]
        return loss

    def loss(self, *args):
        z = self(*args)
        return self.objective.loss(*z)

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
        pred_avg = mean(stack(preds), dim=0)
        losses = [mean(norm(z - pred_avg, dim=0)) for i, z in enumerate(z, start=1)]
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
