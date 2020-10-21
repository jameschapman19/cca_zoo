from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F

import cca_zoo.deep_models

"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""


class DVCCA(nn.Module, ABC):
    """
    https: // arxiv.org / pdf / 1610.03454.pdf
    With pieces borrowed from the variational autoencoder implementation @
    # https: // github.com / pytorch / examples / blob / master / vae / main.py

    A couple of important variables here, both_encoders and private.
    Both_encoders is something I added so that we could potentially compare the effect of using
    Private is as described in the paper and adds another encoder for private information for each view.
    For this reason the hidden dimensions passed to the decoders is 3*latent_dims as we concanate shared,private_1 and private_2
    """

    def __init__(self, input_size_1, input_size_2, hidden_layer_sizes_1=None, hidden_layer_sizes_2=None, latent_dims=2,
                 mu=0.5, both_encoders=True, private=False):
        super(DVCCA, self).__init__()

        self.private = private
        self.both_encoders = both_encoders
        self.mu = mu

        self.encoder_1 = cca_zoo.deep_models.Encoder(hidden_layer_sizes_1, input_size_1, latent_dims).double()
        if self.both_encoders:
            self.encoder_2 = cca_zoo.deep_models.Encoder(hidden_layer_sizes_2, input_size_2, latent_dims).double()

        self.private_encoder_1 = cca_zoo.deep_models.Encoder(hidden_layer_sizes_1, input_size_1, latent_dims).double()
        self.private_encoder_2 = cca_zoo.deep_models.Encoder(hidden_layer_sizes_2, input_size_2, latent_dims).double()

        if self.private:
            self.decoder_1 = cca_zoo.deep_models.Decoder(hidden_layer_sizes_1 * 3, latent_dims, input_size_1).double()
            self.decoder_2 = cca_zoo.deep_models.Decoder(hidden_layer_sizes_2 * 3, latent_dims, input_size_2).double()
        else:
            self.decoder_1 = cca_zoo.deep_models.Decoder(hidden_layer_sizes_1, latent_dims, input_size_1).double()
            self.decoder_2 = cca_zoo.deep_models.Decoder(hidden_layer_sizes_2, latent_dims, input_size_2).double()

    def encode_1(self, x):
        # 2*latent_dims
        z = self.encoder_1(x)
        z = z.reshape((2, -1, self.latent_dims))
        mu = z[0]
        logvar = z[0]
        return mu, logvar

    def encode_2(self, x):
        # 2*latent_dims
        z = self.encoder_2(x)
        z = z.reshape((2, -1, self.latent_dims))
        mu = z[0]
        logvar = z[0]
        return mu, logvar

    def encode_private_1(self, x):
        # 2*latent_dims
        z = self.private_encoder_1(x)
        z = z.reshape((2, -1, self.latent_dims))
        mu = z[0]
        logvar = z[0]
        return mu, logvar

    def encode_private_2(self, x):
        # 2*latent_dims
        z = self.private_encoder_2(x)
        z = z.reshape((2, -1, self.latent_dims))
        mu = z[0]
        logvar = z[0]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Use the standard deviation from the encoder
        std = torch.exp(0.5 * logvar)
        # Mutliply with additive noise (assumed gaussian observation model)
        eps = torch.randn_like(std)
        # Generate random sample
        return mu + eps * std

    def decode_1(self, z):
        # 2*latent_dims
        x = self.decoder_1(z)
        return x

    def decode_2(self, z):
        # 2*latent_dims
        x = self.decoder_2(z)
        return x

    def forward(self, x_1, x_2=None):
        mu_1, logvar_1 = self.encode_1(x_1)
        z_1 = self.reparameterize(mu_1, logvar_1)
        if self.both_encoders:
            mu_2, logvar_2 = self.encode_2(x_2)
            z_2 = self.reparameterize(mu_2, logvar_2)

        if self.private:
            mu_p1, logvar_p1 = self.encode_private_1(x_1)
            z_p1 = self.reparameterize(mu_p1, logvar_p1)
            mu_p2, logvar_p2 = self.encode_private_2(x_2)
            z_p2 = self.reparameterize(mu_p2, logvar_p2)
            z_1_decode = torch.cat([z_1, z_p1, z_p2], dim=-1)
            z_2_decode = torch.cat([z_2, z_p1, z_p2], dim=-1)
        else:
            return self.decode_1(z_1), self.decode_2(z_2), mu_1, logvar_1

        return self.decode_1(z_1_decode), self.decode_2(z_2_decode), mu_1, logvar_1, mu_p1, logvar_p1, mu_p2, logvar_p2

    def loss(self, x_1, x_2, recon_1, recon_2, mu_1, logvar_1, mu_p1, logvar_p1, mu_2=None, logvar_2=None, mu_p2=None,
             logvar_p2=None):

        BCE_1 = F.binary_cross_entropy(recon_1, x_1, reduction='sum')

        BCE_2 = F.binary_cross_entropy(recon_2, x_2, reduction='sum')

        # KL bit - we have assumed logvar diagonal
        KL_1 = -0.5 * torch.sum(1 + logvar_1 - logvar_1.exp() - mu_1.pow(2))
        if self.both_encoders:
            KL_2 = -0.5 * torch.sum(1 + logvar_2 - logvar_2.exp() - mu_2.pow(2))

        if self.private:
            KL_p1 = -0.5 * torch.sum(1 + logvar_p1 - logvar_p1.exp() - mu_p1.pow(2))
            KL_p2 = -0.5 * torch.sum(1 + logvar_p2 - logvar_p2.exp() - mu_p2.pow(2))
            if self.both_encoders:
                return self.mu * KL_1 + (1 - self.mu) * KL_2 + BCE_1 + BCE_2 + KL_p1 + KL_p2
            else:
                return KL_1 + BCE_1 + BCE_2 + KL_p1
        else:
            if self.both_encoders:
                return self.mu * KL_1 + (1 - self.mu) * KL_2 + BCE_1 + BCE_2
            else:
                return KL_1 + BCE_1 + BCE_2
