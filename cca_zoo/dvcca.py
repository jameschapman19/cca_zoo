"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""

from typing import Iterable

import torch
import torch.distributions as dist
from torch.nn import functional as F

from cca_zoo.dcca import _DCCA_base
from cca_zoo.deep_models import BaseEncoder, Encoder, BaseDecoder, Decoder


class DVCCA(_DCCA_base):
    """
    https: // arxiv.org / pdf / 1610.03454.pdf
    With pieces borrowed from the variational autoencoder implementation @
    # https: // github.com / pytorch / examples / blob / master / vae / main.py
    """

    def __init__(self, latent_dims: int, encoders: Iterable[BaseEncoder] = (Encoder, Encoder),
                 decoders: Iterable[BaseDecoder] = (Decoder, Decoder),
                 private_encoders: Iterable[BaseEncoder] = (Encoder, Encoder), learning_rate=1e-3, private=False,
                 mu=0.5, encoder_optimizers=None, decoder_optimizers=None,
                 private_encoder_optimizers=None,
                 encoder_schedulers=None, decoder_schedulers=None, private_encoder_schedulers=None):
        super().__init__(latent_dims)
        self.private = private
        self.mu = mu
        self.latent_dims = latent_dims
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        self.schedulers = []
        if encoder_schedulers:
            self.schedulers.extend(encoder_schedulers)
        if decoder_schedulers:
            self.schedulers.extend(decoder_schedulers)
        self.encoder_optimizers = encoder_optimizers
        if self.encoder_optimizers is None:
            self.encoder_optimizers = torch.optim.Adam(self.encoders.parameters(), lr=learning_rate)
        self.decoder_optimizers = decoder_optimizers
        if self.decoder_optimizers is None:
            self.decoder_optimizers = torch.optim.Adam(self.decoders.parameters(), lr=learning_rate)
        if private:
            self.private_encoders = torch.nn.ModuleList(private_encoders)
            self.private_encoder_optimizers = private_encoder_optimizers
            if self.private_encoder_optimizers is None:
                self.private_encoder_optimizers = torch.optim.Adam(self.private_encoders.parameters(), lr=learning_rate)
            if private_encoder_schedulers:
                self.schedulers.extend(private_encoder_schedulers)

    def update_weights(self, *args):
        """
        :param args:
        :return:
        """
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

    def forward(self, *args, mle=True):
        """
        :param args:
        :param mle:
        :return:
        """
        # Used when we get reconstructions
        mu, logvar = self.encode(*args)
        if mle:
            z = mu
        else:
            z_dist = dist.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()
        # If using single encoder repeat representation n times
        if len(self.encoders) == 1:
            z = z * len(args)
        if self.private:
            mu_p, logvar_p = self.encode_private(*args)
            if mle:
                z_p = mu_p
            else:
                z_dist = dist.Normal(mu_p, torch.exp(0.5 * logvar_p))
                z = z_dist.rsample()
            z = [torch.cat([z_] + z_p, dim=-1) for z_ in z]
        return z

    def encode(self, *args):
        """
        :param args:
        :return:
        """
        mu = []
        logvar = []
        for i, encoder in enumerate(self.encoders):
            mu_i, logvar_i = encoder(args[i])
            mu.append(mu_i)
            logvar.append(logvar_i)
        return mu, logvar

    def encode_private(self, *args):
        """
        :param args:
        :return:
        """
        mu = []
        logvar = []
        for i, private_encoder in enumerate(self.private_encoders):
            mu_i, logvar_i = private_encoder(args[i])
            mu.append(mu_i)
            logvar.append(logvar_i)
        return mu, logvar

    def decode(self, z):
        """
        :param z:
        :return:
        """
        x = []
        for i, decoder in enumerate(self.decoders):
            x_i = decoder(z)
            x.append(x_i)
        return tuple(x)

    def recon(self, *args):
        """
        :param args:
        :return:
        """
        z = self(*args)
        return [self.decode(z_i) for z_i in z][0]

    def loss(self, *args):
        """
        :param args:
        :return:
        """
        mus, logvars = self.encode(*args)
        if self.private:
            mus_p, logvars_p = self.encode_private(*args)
            losses = [self.vcca_private_loss(*args, mu=mu, logvar=logvar, mu_p=mu_p, logvar_p=logvar_p) for
                      (mu, logvar, mu_p, logvar_p) in
                      zip(mus, logvars, mus_p, logvars_p)]
        else:
            losses = [self.vcca_loss(*args, mu=mu, logvar=logvar) for (mu, logvar) in
                      zip(mus, logvars)]
        return torch.stack(losses).mean()

    def vcca_loss(self, *args, mu, logvar):
        """
        :param args:
        :param mu:
        :param logvar:
        :return:
        """
        batch_n = mu.shape[0]
        z_dist = dist.Normal(mu, torch.exp(0.5 * logvar))
        z = z_dist.rsample()
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
        recons = self.decode(z)
        bces = torch.stack(
            [F.binary_cross_entropy(recon, arg, reduction='sum') / batch_n for recon, arg in
             zip(recons, args)]).sum()
        return kl + bces

    def vcca_private_loss(self, *args, mu, logvar, mu_p, logvar_p):
        """
        :param args:
        :param mu:
        :param logvar:
        :return:
        """
        batch_n = mu.shape[0]
        z_dist = dist.Normal(mu, torch.exp(0.5 * logvar))
        z = z_dist.rsample()
        z_p_dist = dist.Normal(mu_p, torch.exp(0.5 * logvar_p))
        z_p = z_p_dist.rsample()
        kl_p = torch.stack(
            [torch.mean(-0.5 * torch.sum(1 + logvar_p - logvar_p.exp() - mu_p.pow(2), dim=1), dim=0) for
             i, _ in enumerate(self.private_encoders)]).sum()
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
        z_combined = torch.cat([z, z_p], dim=-1)
        recon = self.decode(z_combined)
        bces = torch.stack(
            [F.binary_cross_entropy(recon[i], args[i], reduction='sum') / batch_n for i, _ in
             enumerate(self.decoders)]).sum()
        return kl + kl_p + bces
