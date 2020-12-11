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
                                       config.latent_dims, variational=True).double()
    return encoder


def create_private_encoder(config, i):
    encoder = config.private_encoder_models[i](config.hidden_layer_sizes[i], config.input_sizes[i],
                                               config.latent_dims, variational=True).double()
    return encoder


def create_decoder(config, i):
    decoder = config.decoder_models[i](config.hidden_layer_sizes[i], config.latent_dims, config.input_sizes[i]).double()
    return decoder


def create_private_decoder(config, i):
    decoder = config.decoder_models[i](config.hidden_layer_sizes[i], config.latent_dims * 3,
                                       config.input_sizes[i]).double()
    return decoder


class DVCCA(nn.Module):
    """
    https: // arxiv.org / pdf / 1610.03454.pdf
    With pieces borrowed from the variational autoencoder implementation @
    # https: // github.com / pytorch / examples / blob / master / vae / main.py
    """

    def __init__(self, config: Config = Config):
        super(DVCCA, self).__init__()
        self.private = config.private
        self.mu = config.mu
        self.latent_dims = config.latent_dims
        self.encoders = nn.ModuleList([create_encoder(config, i) for i in range(len(config.encoder_models))])
        if config.private:
            self.private_encoders = nn.ModuleList(
                [create_private_encoder(config, i) for i in range(len(config.private_encoder_models))])
        if config.private:
            self.decoders = nn.ModuleList(
                [create_private_decoder(config, i) for i in range(len(config.decoder_models))])
        else:
            self.decoders = nn.ModuleList([create_decoder(config, i) for i in range(len(config.decoder_models))])

        self.encoder_optimizers = optim.Adam(self.encoders.parameters(), lr=config.learning_rate)
        self.decoder_optimizers = optim.Adam(self.decoders.parameters(), lr=config.learning_rate)
        if self.private:
            self.private_encoder_optimizers = optim.Adam(self.private_encoders.parameters(), lr=config.learning_rate)

    def encode(self, *args):
        mu = []
        logvar = []
        for i, encoder in enumerate(self.encoders):
            mu_i, logvar_i = encoder(args[i])
            mu.append(mu_i)
            logvar.append(logvar_i)
        return mu, logvar

    def encode_private(self, *args):
        mu = []
        logvar = []
        for i, private_encoder in enumerate(self.private_encoders):
            mu_i, logvar_i = private_encoder(args[i])
            mu.append(mu_i)
            logvar.append(logvar_i)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        # Use the standard deviation from the encoder
        std = torch.exp(0.5 * logvar)
        # Mutliply with additive noise (assumed gaussian observation model)
        eps = torch.randn_like(std)
        # Generate random sample
        return mu + eps * std

    def decode(self, z):
        x = []
        for i, decoder in enumerate(self.decoders):
            x_i = torch.sigmoid(decoder(z))
            x.append(x_i)
        return tuple(x)

    def forward(self, *args, mle=True):
        # Used when we get reconstructions
        mu, logvar = self.encode(*args)
        if mle:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        # If using single encoder repeat representation n times
        if len(self.encoders) == 1:
            z = z * len(args)
        if self.private:
            mu_p, logvar_p = self.encode_private(*args)
            if mle:
                z_p = mu_p
            else:
                z_p = self.reparameterize(mu_p, logvar_p)
            z = [torch.cat([z_] + z_p, dim=-1) for z_ in z]
        return z

    def recon(self, *args):
        z = self(*args)
        return [self.decode(z_i) for z_i in z][0]

    def update_weights(self, *args):
        self.encoder_optimizers.zero_grad()
        self.decoder_optimizers.zero_grad()
        if self.private:
            self.private_encoder_optimizers.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        self.encoder_optimizers.step()
        self.decoder_optimizers.step()
        if self.private:
            self.private_encoder_optimizers.step()
        return loss

    def loss(self, *args):
        mu, logvar = self.encode(*args)
        if self.private:
            losses = [self.vcca_private_loss(*args, mu=mu, logvar=logvar[i]) for i, mu in
                      enumerate(mu)]
        else:
            losses = [self.vcca_loss(*args, mu=mu, logvar=logvar[i]) for i, mu in
                      enumerate(mu)]
        return torch.stack(losses).mean()

    def vcca_loss(self, *args, mu, logvar):
        batch_n=mu.shape[0]
        z = self.reparameterize(mu, logvar)
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
        recon = self.decode(z)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum')/batch_n for i, _ in enumerate(self.decoders)]).sum()
        return kl + bces

    def vcca_private_loss(self, *args, mu, logvar):
        batch_n = mu.shape[0]
        z = self.reparameterize(mu, logvar)
        mu_p, logvar_p = self.encode_private(*args)
        z_p = [self.reparameterize(mu_p[i], logvar_p[i]) for i, _ in enumerate(self.private_encoders)]
        kl_p = torch.stack(
            [torch.mean(-0.5 * torch.sum(1 + logvar_p[i] - logvar_p[i].exp() - mu_p[i].pow(2), dim=1), dim=0) for
             i, _ in enumerate(self.private_encoders)]).sum()
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
        z_combined = torch.cat([z] + z_p, dim=-1)
        recon = self.decode(z_combined)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum')/batch_n for i, _ in enumerate(self.decoders)]).sum()
        return kl + kl_p + bces
