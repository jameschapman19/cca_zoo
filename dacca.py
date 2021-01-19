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
    encoder = config.encoder_models[i](config.input_sizes[i], config.latent_dims,
                                       **config.encoder_args[i])
    return encoder


def create_private_encoder(config, i):
    encoder = config.private_encoder_models[i](config.input_sizes[i], config.latent_dims,
                                               **config.private_encoder_args[i])
    return encoder


def create_decoder(config, i):
    decoder = config.decoder_models[i](config.latent_dims, config.input_sizes[i], **config.decoder_args[i])
    return decoder

# SLightly different decode if private encoders are added. This is because we have the extra dimensionality of the 2 private encoders. May need extending for more than 2 views.
def create_private_decoder(config, i):
    decoder = config.decoder_models[i](config.latent_dims * 3, config.input_sizes[i], **config.decoder_args[i])
    return decoder

def create_discriminator(config, i):
    decoder = config.discriminator_models[i](config.latent_dims, 1, **config.decoder_args[i])
    return decoder

class DACCA(nn.Module):
    """
    https: // arxiv.org / pdf / 1610.03454.pdf
    With pieces borrowed from the variational autoencoder implementation @
    # https: // github.com / pytorch / examples / blob / master / vae / main.py
    """

    def __init__(self, config: Config = Config):
        super(DACCA, self).__init__()
        self.samples=5
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
        self.discriminators = nn.ModuleList([create_discriminator(config, i) for i in range(len(config.discriminator_models))])
        self.encoder_optimizers = optim.Adam(self.encoders.parameters(), lr=config.learning_rate)
        self.decoder_optimizers = optim.Adam(self.decoders.parameters(), lr=config.learning_rate)
        self.discriminator_optimizers = optim.Adam(self.discriminators.parameters(), lr=config.learning_rate)
        if self.private:
            self.private_encoder_optimizers = optim.Adam(self.private_encoders.parameters(), lr=config.learning_rate)

    def encode(self, *args):
        z = []
        for i, encoder in enumerate(self.encoders):
            z_i = encoder(args[i])
            z.append(z_i)
        return z

    def encode_private(self, *args):
        z = []
        for i, private_encoder in enumerate(self.private_encoders):
            z_i = private_encoder(args[i])
            z.append(z_i)
        return z

    def decode(self, z):
        x = []
        for i, decoder in enumerate(self.decoders):
            x_i = torch.sigmoid(decoder(z))
            x.append(x_i)
        return tuple(x)

    def forward(self, *args):
        # Used when we get reconstructions
        z = self.encode(*args)
        # If using single encoder repeat representation n times
        if len(self.encoders) == 1:
            z = z * len(args)
        if self.private:
            z_p = self.encode_private(*args)
            z = [torch.cat([z_] + z_p, dim=-1) for z_ in z]
        return z

    def recon(self, *args):
        z = self(*args)
        return [self.decode(z_i) for z_i in z][0]

    def update_weights(self, *args):
        #autoencoder step
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
        #discriminator step
        self.discriminator_optimizers.zero_grad()
        loss = self.discriminator_loss(*args)
        loss.backward()
        self.discriminator_optimizers.step()
        #generator step
        self.encoder_optimizers.zero_grad()
        loss = self.generator_loss(*args)
        loss.backward()
        self.encoder_optimizers.step()
        return loss

    def recon_loss(self, *args):
        z = self.encode(*args)
        if self.private:
            losses = [self.acca_private_recon_loss(*args, z=z) for i, z in
                      enumerate(z)]
        else:
            losses = [self.acca_recon_loss(*args, z=z) for i, z in
                      enumerate(z)]
        return torch.stack(losses).mean()

    def discriminator_loss(self, *args):
        z = self.encode(*args)
        if self.private:
            losses = [self.acca_private_discriminator_loss(*args, z=z) for i, z in
                      enumerate(z)]
        else:
            losses = [self.acca_discriminator_loss(*args, z=z) for i, z in
                      enumerate(z)]
        return torch.stack(losses).mean()

    def generator_loss(self, *args):
        z = self.encode(*args)
        if self.private:
            losses = [self.acca_private_generator_loss(*args, z=z) for i, z in
                      enumerate(z)]
        else:
            losses = [self.acca_generator_loss(*args, z=z) for i, z in
                      enumerate(z)]
        return torch.stack(losses).mean()

    def acca_recon_loss(self, *args, z):
        batch_n = z.shape[0]
        recon = self.decode(z)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum') / batch_n for i, _ in
             enumerate(self.decoders)]).sum()
        return bces

    def acca_private_recon_loss(self, *args, z):
        batch_n = z.shape[0]
        z_p = self.encode_private(*args)
        z_combined = torch.cat([z] + z_p, dim=-1)
        recon = self.decode(z_combined)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum') / batch_n for i, _ in
             enumerate(self.decoders)]).sum()
        return bces

    def acca_discriminator_loss(self, *args, z):
        batch_n = z.shape[0]
        recon = self.decode(z)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum') / batch_n for i, _ in
             enumerate(self.decoders)]).sum()
        return bces

    def acca_private_discriminator_loss(self, *args, z):
        batch_n = z.shape[0]
        z_p = self.encode_private(*args)
        z_combined = torch.cat([z] + z_p, dim=-1)
        recon = self.decode(z_combined)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum') / batch_n for i, _ in
             enumerate(self.decoders)]).sum()
        return bces

    def acca_generator_loss(self, *args, z):
        batch_n = z.shape[0]
        recon = self.decode(z)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum') / batch_n for i, _ in
             enumerate(self.decoders)]).sum()
        return bces

    def acca_private_generator_loss(self, *args, z):
        batch_n = z.shape[0]
        z_p = self.encode_private(*args)
        z_combined = torch.cat([z] + z_p, dim=-1)
        recon = self.decode(z_combined)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum') / batch_n for i, _ in
             enumerate(self.decoders)]).sum()
        return bces