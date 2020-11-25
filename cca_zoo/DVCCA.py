from abc import ABC

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
    encoder = config.encoder_models[i](config.hidden_layer_sizes[i], config.input_sizes[i],
                                       config.latent_dims * 2).double()
    return encoder


def create_decoder(config, i):
    decoder = config.decoder_models[i](config.hidden_layer_sizes[i], config.input_sizes[i], config.latent_dims).double()
    return decoder


def create_private_decoder(config, i):
    decoder = config.decoder_models[i](config.hidden_layer_sizes[i], config.input_sizes[i],
                                       config.latent_dims * 2).double()
    return decoder


class DVCCA(nn.Module):
    """
    https: // arxiv.org / pdf / 1610.03454.pdf
    With pieces borrowed from the variational autoencoder implementation @
    # https: // github.com / pytorch / examples / blob / master / vae / main.py

    A couple of important variables here, both_encoders and private.
    Both_encoders is something I added so that we could potentially compare the effect of using
    Private is as described in the paper and adds another encoder for private information for each view.
    For this reason the hidden dimensions passed to the decoders is 3*latent_dims as we concanate shared,private_1 and private_2
    """

    def __init__(self, config: Config = Config):
        super(DVCCA, self).__init__()
        views = len(config.decoder_models)
        self.private = config.private
        self.both_encoders = config.both_encoders
        self.mu = config.mu
        self.latent_dims = config.latent_dims
        self.encoders = [create_encoder(config, i) for i in range(views)]
        if config.private:
            self.private_encoders = [create_encoder(config, i) for i in range(views)]

        if config.private:
            self.decoders = [create_private_decoder(config, i) for i in range(views)]
        else:
            self.decoders = [create_decoder(config, i) for i in range(views)]

        self.encoder_optimizers = [optim.Adam(list(encoder.parameters()), lr=config.learning_rate) for encoder in self.encoders]
        self.decoder_optimizers = [optim.Adam(list(decoder.parameters()), lr=config.learning_rate) for decoder in self.decoders]
        if self.private:
            self.private_encoder_optimizers = [optim.Adam(list(encoder.parameters()), lr=config.learning_rate) for encoder in self.private_encoders]

    def encode(self, *args):
        mu = []
        logvar = []
        for i, encoder in enumerate(self.encoders):
            z = encoder(args[i])
            z = z.reshape((2, -1, self.latent_dims))
            mu.append(z[0])
            logvar.append(z[1])
        return mu, logvar

    def encode_private(self, *args):
        mu = []
        logvar = []
        for i, private_encoder in enumerate(self.private_encoders):
            z = private_encoder(args[i])
            z = z.reshape((2, -1, self.latent_dims))
            mu.append(z[0])
            logvar.append(z[1])
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        # Use the standard deviation from the encoder
        std = torch.exp(0.5 * logvar)
        # Mutliply with additive noise (assumed gaussian observation model)
        eps = torch.randn_like(std)
        # Generate random sample
        return mu + eps * std

    def decode(self, *args):
        x = []
        for i, decoder in enumerate(self.decoders):
            x_i = decoder(args[i])
            x.append(x_i)
        return tuple(x)

    def forward(self, *args):
        # Used when we get reconstructions
        mu, logvar = self.encode(*args)
        z = mu
        return z

    def recon(self, *args):
        # Used when we get reconstructions
        mu, logvar = self.encode(*args)
        z = mu
        # If using single encoder repeat representation n times
        if len(self.encoders) == 1:
            z = z * len(args)
        if self.private:
            mu_p, logvar_p = self.encode_private(*args)
            z_p = mu_p
            z = [torch.cat([z[i], z_p[i]], dim=-1) for i, _ in enumerate(args)]
        return self.decode(z)

    def update_weights(self, *args):
        [optimizer.zero_grad() for optimizer in self.encoder_optimizers]
        [optimizer.zero_grad() for optimizer in self.decoder_optimizers]
        if self.private:
            [optimizer.zero_grad() for optimizer in self.private_encoder_optimizers]
        loss = self.loss(*args)
        loss.backward()
        [optimizer.step() for optimizer in self.encoder_optimizers]
        [optimizer.step() for optimizer in self.decoder_optimizers]
        if self.private:
            [optimizer.step() for optimizer in self.private_encoder_optimizers]
        return loss

    def loss(self, *args):
        mu, logvar = self.encode(*args)
        z = [self.reparameterize(mu[i], logvar[i]) for i, _ in enumerate(self.encoders)]
        kl = torch.stack([-0.5 * torch.sum(1 + logvar[i] - logvar[i].exp() - mu[i].pow(2)) for i, _ in
                          enumerate(self.encoders)]).sum(dim=0)
        if len(self.encoders) == 1:
            z = z * len(args)
        if self.private:
            mu_p, logvar_p = self.encode_private(*args)
            z_p = [self.reparameterize(mu_p[i], logvar_p[i]) for i, _ in enumerate(self.private_encoders)]
            z = [torch.cat([z[i], z_p[i]], dim=-1) for i, _ in enumerate(args)]

        recon = self.decode(*z)

        # LOSS
        bce = torch.stack([F.mse_loss(recon[i], x, reduction='sum') for i, x in enumerate(args)]).sum(dim=0)/len(args)

        if self.private:
            kl_p = torch.stack([-0.5 * torch.sum(1 + logvar_p[i] - logvar_p[i].exp() - mu_p[i].pow(2)) for i, _ in
                                enumerate(self.private_encoders)]).sum(dim=0)
            return (kl + kl_p + bce)/args[0].shape[0]
        else:
            return (kl + bce)/args[0].shape[0]
