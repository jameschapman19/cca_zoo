import torch
import torch.nn.functional as F

from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder, Decoder
from cca_zoo.deepmodels.dcca import _DCCA_base


class SplitAE(_DCCA_base):
    """
    A class used to fit a Split Autoencoder model.

    :Citation:

    Ngiam, Jiquan, et al. "Multimodal deep learning." ICML. 2011.

    """

    def __init__(self, latent_dims: int, encoder: BaseEncoder = Encoder, decoders=None):
        """

        :param latent_dims: # latent dimensions
        :param encoder: list of encoder networks
        :param decoders:  list of decoder networks
        """
        super().__init__(latent_dims=latent_dims)
        if decoders is None:
            decoders = [Decoder, Decoder]
        self.encoder = encoder
        self.decoders = torch.nn.ModuleList(decoders)

    def forward(self, *args):
        z = self.encoder(args[0])
        return [z]

    def recon(self, *args):
        """
        :param args:
        :return:
        """
        z = self(*args)
        return self._decode(z)

    def _decode(self, *z):
        """
        This method is used to decode from the latent space to the best prediction of the original views

        :param z:
        """
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(*z))
        return tuple(recon)

    def loss(self, *args):
        z = self(*args)
        recon = self._decode(*z)
        recon_loss = self.recon_loss(args, recon)
        return recon_loss

    @staticmethod
    def recon_loss(x, recon):
        recons = [
            F.mse_loss(recon[i], x[i], reduction="mean") for i in range(len(recon))
        ]
        return torch.stack(recons).sum(dim=0)
