import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from cca_zoo.configuration import Config

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


def create_decoder(config, i):
    decoder = config.decoder_models[i](config.hidden_layer_sizes[i], config.latent_dims, config.input_sizes[i]).double()
    return decoder


class DCCAE(nn.Module):

    def __init__(self, config: Config = Config):
        super(DCCAE, self).__init__()
        views = len(config.encoder_models)
        self.encoders = torch.nn.ModuleList([create_encoder(config, i) for i in range(views)])
        self.decoders = torch.nn.ModuleList([create_decoder(config, i) for i in range(views)])
        self.lam = config.lam
        self.objective = config.objective(config.latent_dims)
        self.optimizers = [optim.Adam(list(self.encoders[i].parameters()) + list(self.decoders[i].parameters()),
                                      lr=config.learning_rate) for i in range(views)]

    def encode(self, *args):
        z = []
        for i, arg in enumerate(args):
            z.append(self.encoders[i](arg))
        return tuple(z)

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def decode(self, *args):
        recon = []
        for i, arg in enumerate(args):
            recon.append(self.decoders[i](arg))
        return tuple(recon)

    def update_weights(self, *args):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        loss = self.loss(*args)
        loss.backward()
        [optimizer.step() for optimizer in self.optimizers]
        return loss

    def loss(self, *args):
        z = self.encode(*args)
        recon = self.decode(*z)
        recon_loss = self.recon_loss(args, recon)
        return self.lam * recon_loss + self.objective.loss(*z)

    @staticmethod
    def recon_loss(x, recon):
        recons = [F.mse_loss(recon[i], x[i], reduction='sum') for i in range(len(x))]
        return torch.stack(recons).sum(dim=0)
