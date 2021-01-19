"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""

from torch import nn
from torch.linalg import norm
from torch import optim, matmul, mean, stack
from torch.nn import functional as F
from cca_zoo.objectives import compute_matrix_power
from abc import abstractmethod


class DCCA(nn.Module):
    def __init__(self, objective=None, input_sizes=None, latent_dims=1, encoder_models=None, encoder_args=None,
                 learning_rate=1e-3, als=False):
        super(DCCA, self).__init__()
        self.encoders = nn.ModuleList(
            [model(input_sizes[i], latent_dims, **encoder_args[i]) for i, model in
             enumerate(encoder_models)])
        self.objective = objective(latent_dims)
        self.optimizers = [optim.Adam(list(encoder.parameters()), lr=learning_rate) for encoder in self.encoders]
        self.covs = None
        if als:
            self.update_weights = self.update_weights_als
        else:
            self.update_weights = self.update_weights_tn
            self.loss = self.tn_loss

    @abstractmethod
    def update_weights(self, *args):
        if self.als:
            loss = self.update_weights_als(*args)
        else:
            loss = self.update_weights_tn(*args)
        return loss

    @abstractmethod
    def forward(self, *args):
        """
        :param args:
        :return:
        """
        z = self.encode(*args)
        return z

    def encode(self, *args):
        """
        :param args:
        :return:
        """
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return tuple(z)

    def update_weights_tn(self, *args):
        """
        :param args:
        :return:
        """
        [optimizer.zero_grad() for optimizer in self.optimizers]
        loss = self.tn_loss(*args)
        loss.backward()
        [optimizer.step() for optimizer in self.optimizers]
        return loss

    def tn_loss(self, *args):
        """
        :param args:
        :return:
        """
        z = self(*args)
        return self.objective.loss(*z)

    def update_weights_als(self, *args):
        """
        :param args:
        :return:
        """
        losses, obj = self.als_loss(*args)
        self.optimizers[0].zero_grad()
        losses[0].backward()
        self.optimizers[0].step()
        self.optimizers[1].zero_grad()
        losses[1].backward()
        self.optimizers[1].step()
        return obj  # sum(losses) / 2 - self.config.latent_dims

    def als_loss(self, *args):
        """
        :param args:
        :return:
        """
        z = self(*args)
        self.update_covariances(*z)
        covariance_inv = [compute_matrix_power(cov, -0.5, self.config.eps) for cov in self.covs]
        preds = [matmul(z, covariance_inv[i]).detach() for i, z in enumerate(z)]
        pred_avg = mean(stack(preds), dim=0)
        losses = [mean(norm(z - pred_avg, dim=0)) for i, z in enumerate(z, start=1)]
        obj = self.objective.loss(*z)
        return losses, obj

    def update_covariances(self, *args):
        """
        :param args:
        :return:
        """
        b = args[0].shape[0]
        batch_covs = [z_i.T @ z_i for i, z_i in enumerate(args)]
        if self.covs is not None:
            self.covs = [(self.config.rho * self.covs[i]).detach() + (1 - self.config.rho) * batch_cov for i, batch_cov
                         in
                         enumerate(batch_covs)]
        else:
            self.covs = batch_covs


