from torch import nn
from torch import optim, matmul
from torch.nn import functional as F

from cca_zoo.configuration import Config
from cca_zoo.objectives import compute_matrix_power

"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""


def create_encoder(config, i):
    encoder = config.encoder_models[i](config.hidden_layer_sizes[i], config.input_sizes[i], config.latent_dims).double()
    return encoder


class DCCA(nn.Module):

    def __init__(self, config: Config = Config):
        super(DCCA, self).__init__()
        views = len(config.encoder_models)
        self.config = config
        self.encoders = nn.ModuleList([create_encoder(config, i) for i in range(views)])
        self.objective = config.objective(config.latent_dims)
        self.optimizers = [optim.Adam(list(encoder.parameters()), lr=config.learning_rate) for encoder in self.encoders]
        self.covs = None
        if config.als:
            self.update_weights = self.update_weights_als
            self.loss = self.als_loss_validation
        else:
            self.update_weights = self.update_weights_tn
            self.loss = self.tn_loss

    def encode(self, *args):
        z = []
        for i, arg in enumerate(args):
            z.append(self.encoders[i](arg))
        return tuple(z)

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def update_weights_tn(self, *args):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        loss = self.tn_loss(*args)
        loss.backward()
        [optimizer.step() for optimizer in self.optimizers]
        return loss

    def tn_loss(self, *args):
        z = self(*args)
        return self.objective.loss(*z)

    def update_weights_als(self, *args):
        loss_1, loss_2 = self.als_loss(*args)
        self.optimizers[0].zero_grad()
        loss_1.backward()
        self.optimizers[0].step()
        self.optimizers[1].zero_grad()
        loss_2.backward()
        self.optimizers[1].step()
        return (loss_1 + loss_2) / 2 - self.config.latent_dims

    def als_loss(self, *args):
        z = self(*args)
        self.update_covariances(*z)
        covariance_inv = [compute_matrix_power(cov, -0.5, self.config.eps) for cov in self.covs]
        preds = [matmul(z, covariance_inv[i]).detach() for i, z in enumerate(z)]
        # Least squares for each projection in same manner as linear from before
        # Currently 2 view case
        losses = [F.mse_loss(preds[-i], z) for i, z in enumerate(z)]
        return losses

    def als_loss_validation(self, *args):
        z = self(*args)
        SigmaHat11RootInv = compute_matrix_power(self.covs[0], -0.5, self.config.eps)
        SigmaHat22RootInv = compute_matrix_power(self.covs[1], -0.5, self.config.eps)
        pred_1 = (z[0] @ SigmaHat11RootInv).detach()
        pred_2 = (z[1] @ SigmaHat22RootInv).detach()
        # Least squares for each projection in same manner as linear from before
        loss_1 = F.mse_loss(pred_1, z[1])
        loss_2 = F.mse_loss(pred_2, z[0])
        return (loss_1 + loss_2) / 2 - self.config.latent_dims

    def update_covariances(self, *args):
        b = args[0].shape[0]
        batch_covs = [z_i.T @ z_i / b for i, z_i in enumerate(args)]
        if self.covs is not None:
            self.covs = [(self.config.rho * self.covs[i]).detach() + (1 - self.config.rho) * batch_cov for i, batch_cov
                         in
                         enumerate(batch_covs)]
        else:
            self.covs = batch_covs
