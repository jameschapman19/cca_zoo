from abc import ABC

from torch import nn
from torch import optim
from torch import inverse


from cca_zoo.configuration import Config

"""
Non-linear Deconfounding Deep CCA

All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""
def create_encoder(config, i):
    encoder = config.encoder_models[i](config.hidden_layer_sizes[i], config.input_sizes[i], config.latent_dims).double()
    return encoder

class DeCCA(nn.Module):
    def __init__(self, config: Config = Config):
        super(DeCCA, self).__init__()
        views = len(config.encoder_models)
        confounds = len(config.confound_encoder_models)
        self.config = config
        self.encoders = [create_encoder(config, i) for i in range(views)]
        self.confound_encoders = [create_encoder(config, i) for i in range(confounds)]
        self.objective = config.objective(config.latent_dims)
        self.optimizers = [optim.Adam(list(encoder.parameters()), lr=config.learning_rate) for encoder in self.encoders]
        self.confound_optimizers = [optim.Adam(list(confound_encoder.parameters()), lr=config.learning_rate) for confound_encoder in self.confound_encoders]

    def encode(self, x_1, x_2, x_c):
        z_1 = self.encoder_1(x_1)
        z_2 = self.encoder_2(x_2)
        z_c = self.encoder_c(x_c)
        C = z_c @ inverse(z_c.T @ z_c) @ z_c.T
        z_1 = z_1 - C @ z_1
        z_2 = z_2 - C @ z_2
        return z_1, z_2, z_c

    def forward(self, x_1, x_2, x_c):
        z_1, z_2, z_c = self.encode(x_1, x_2, x_c)
        return z_1, z_2

    def update_weights(self, x_1, x_2, x_c):
        self.optimizer.zero_grad()
        loss = self.loss(x_1, x_2, x_c)
        loss.backward()
        self.optimizer.step()
        self.optimizer_c.zero_grad()
        confound_loss = -self.loss(x_1, x_2, x_c)
        confound_loss.backward()
        self.optimizer_c.step()
        return loss

    def loss(self, x_1, x_2, x_c):
        z_1, z_2, z_c = self.encode(x_1, x_2, x_c)
        cca = self.cca_objective.loss(z_1, z_2)
        return cca

    def loss_confounds(self, x_1, x_2, x_c):
        z_1, z_2, z_c = self.encode(x_1, x_2, x_c)
        cca_confounds_1 = self.cca_objective.loss(z_1, z_c)
        cca_confounds_2 = self.cca_objective.loss(z_2, z_c)
        return (cca_confounds_1 + cca_confounds_2)/2
