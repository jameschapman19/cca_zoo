from typing import Iterable

import torch
import torch.distributions as dist
from torch.nn import functional as F

from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder, Decoder
from cca_zoo.deepmodels.dcca import _DCCA_base


class DVCCA(_DCCA_base):
    """
    A class used to fit a DVCCA model.

    https: // arxiv.org / pdf / 1610.03454.pdf
    With pieces borrowed from the variational autoencoder implementation @
    # https: // github.com / pytorch / examples / blob / master / vae / main.py
    """

    def __init__(self, latent_dims: int, encoders=None,
                 decoders=None, private_encoders: Iterable[BaseEncoder] = None):
        """
        :param latent_dims: # latent dimensions
        :param encoders: list of encoder networks
        :param decoders:  list of decoder networks
        :param private_encoders: list of private (view specific) encoder networks
        """
        super().__init__(latent_dims=latent_dims)
        if decoders is None:
            decoders = [Decoder, Decoder]
        if encoders is None:
            encoders = [Encoder, Encoder]
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        if private_encoders:
            self.private_encoders = torch.nn.ModuleList(private_encoders)
        else:
            self.private_encoders = None

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
        if self.private_encoders:
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
        if self.private_encoders:
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
