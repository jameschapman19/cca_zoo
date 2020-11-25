from torch import nn
from torch import optim
from torch import inverse
from cca_zoo.configuration import Config

"""
Linear Deconfounding Deep CCA

All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""
def create_encoder(config, i):
    encoder = config.encoder_models[i](config.hidden_layer_sizes[i], config.input_sizes[i], config.latent_dims).double()
    return encoder


class DeLCCA(nn.Module):

    def __init__(self, config: Config = Config):
        super(DeLCCA, self).__init__()
        views = len(config.encoder_models)
        self.config = config
        self.encoders = [create_encoder(config, i) for i in range(views)]
        self.objective = config.objective(config.latent_dims)
        self.optimizers = [optim.Adam(list(encoder.parameters()), lr=config.learning_rate) for encoder in self.encoders]

    def encode(self, *args, x_c=None):
        z = []
        for i, arg in enumerate(args):
            z.append(self.encoders[i](arg))
        C = x_c @ inverse(x_c.T @ x_c) @ x_c.T
        z = [z_i - C @ z_i for z_i in z]
        return z

    def forward(self, *args, x_c=None):
        z_1, z_2 = self.encode(x_1, x_2, x_c)
        return z_1, z_2

    def update_weights(self, x_1, x_2, x_c):
        self.optimizer.zero_grad()
        loss = self.loss(x_1, x_2, x_c)
        loss.backward()
        self.optimizer.step()
        return loss

    def loss(self, x_1, x_2, x_c):
        z_1, z_2 = self.encode(x_1, x_2, x_c)
        cca = self.cca_objective.loss(z_1, z_2)
        return cca
