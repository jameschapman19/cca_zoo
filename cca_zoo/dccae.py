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
    encoder = config.encoder_models[i](config.input_sizes[i], config.latent_dims, **config.encoder_args[i])
    return encoder


def create_decoder(config, i):
    decoder = config.decoder_models[i](config.latent_dims, config.input_sizes[i], **config.decoder_args[i])
    return decoder


class DCCAE(nn.Module):

    def __init__(self, config: Config = Config):
        super(DCCAE, self).__init__()
        self.encoders = torch.nn.ModuleList(
            [create_encoder(config, i) for i, model in enumerate(config.encoder_models)])
        self.decoders = torch.nn.ModuleList(
            [create_decoder(config, i) for i, model in enumerate(config.decoder_models)])
        self.lam = config.lam
        self.objective = config.objective(config.latent_dims)
        self.optimizer = optim.Adam(list(self.encoders.parameters()) + list(self.decoders.parameters()),lr=config.learning_rate)

    def encode(self, *args):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return tuple(z)

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def decode(self, *args):
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(args[i]))
        return tuple(recon)

    def update_weights(self, *args):
        self.optimizer.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        self.optimizer.step()
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
