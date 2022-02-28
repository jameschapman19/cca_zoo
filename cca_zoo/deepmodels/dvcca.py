from typing import Iterable

import torch
import torch.distributions as dist
from torch.nn import functional as F

from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder, Decoder
from cca_zoo.deepmodels.dcca import _DCCA_base


class DVCCA(_DCCA_base):
    """
    A class used to fit a DVCCA model.

    :Citation:

    Wang, Weiran, et al. "Deep variational canonical correlation analysis." arXiv preprint arXiv:1610.03454 (2016).

    https: // arxiv.org / pdf / 1610.03454.pdf

    https: // github.com / pytorch / examples / blob / master / vae / main.py

    """

    def __init__(
        self,
        latent_dims: int,
        encoders=None,
        decoders=None,
        private_encoders: Iterable[BaseEncoder] = None,
    ):
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
        z = dict()
        mu = dict()
        logvar = dict()
        # Used when we get reconstructions
        mu["shared"], logvar["shared"] = self._encode(*args)
        z["shared"] = self._sample(mu["shared"], logvar["shared"], mle)
        # If using single encoder repeat representation n times
        if len(self.encoders) == 1:
            z["shared"] = z["shared"] * len(args)
        if self.private_encoders:
            mu["private"], logvar["private"] = self._encode_private(*args)
            z["private"] = self._sample(mu["private"], logvar["private"], mle)
        return z, mu, logvar

    def _sample(self, mu, logvar, mle):
        """

        :param mu:
        :param logvar:
        :param mle: whether to return the maximum likelihood (i.e. mean) or whether to sample
        :return: a sample from latent vector
        """
        if mle:
            return mu
        else:
            return [
                dist.Normal(mu_, torch.exp(0.5 * logvar_)).rsample()
                for mu_, logvar_ in zip(mu, logvar)
            ]

    def _encode(self, *args):
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

    def _encode_private(self, *args):
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

    def _decode(self, z):
        """
        :param z:
        :return:
        """
        x = []
        for i, decoder in enumerate(self.decoders):
            if "private" in z:
                x_i = F.sigmoid(
                    decoder(torch.cat((z["shared"][i], z["private"][i]), dim=-1))
                )
            else:
                x_i = F.sigmoid(decoder(z["shared"][i]))
            x.append(x_i)
        return x

    def recon(self, *args, mle=True):
        """
        :param args:
        :return:
        """
        z, _, _ = self(*args, mle=mle)
        return self._decode(z)

    def loss(self, *args):
        """
        :param args:
        :return:
        """
        z, mu, logvar = self(*args, mle=False)
        loss = dict()
        loss["reconstruction"] = self.recon_loss(args, z)
        loss["kl shared"] = self.kl_loss(mu["shared"], logvar["shared"])
        if "private" in z:
            loss["kl private"] = self.kl_loss(mu["private"], logvar["private"])
        loss["objective"] = torch.stack(tuple(loss.values())).sum()
        return loss

    @staticmethod
    def kl_loss(mu, logvar):
        return torch.stack(
            [
                torch.mean(
                    -0.5 * torch.sum(1 + logvar_ - logvar_.exp() - mu_.pow(2), dim=1),
                    dim=0,
                )
                for mu_, logvar_ in zip(mu, logvar)
            ]
        ).sum()

    def recon_loss(self, x, z):
        recon = self._decode(z)
        return torch.stack(
            [
                F.binary_cross_entropy(recon, arg, reduction="mean")
                for recon, arg in zip(recon, x)
            ]
        ).sum()
