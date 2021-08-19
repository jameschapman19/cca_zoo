import torch
import torch.nn.functional as F

from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder, Decoder
from cca_zoo.deepmodels.dcca import _DCCA_base


class SplitAE(_DCCA_base):
    """
    A class used to fit a Split Autoencoder model.

    Examples
    --------
    >>> from cca_zoo.deepmodels import SplitAE
    >>> model = SplitAE()
    """

    def __init__(self, latent_dims: int, encoder: BaseEncoder = Encoder,
                 decoders=None):
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
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = self.encoder(args[0])
        return z

    def decode(self, z):
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(z))
        return tuple(recon)

    def loss(self, *args):
        z = self.encode(*args)
        recon = self.decode(z)
        recon_loss = self.recon_loss(args, recon)
        return recon_loss

    @staticmethod
    def recon_loss(x, recon):
        recons = [F.mse_loss(recon[i], x[i], reduction='mean') for i in range(len(x))]
        return torch.stack(recons).sum(dim=0)
