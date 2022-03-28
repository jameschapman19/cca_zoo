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

    def __init__(
        self,
        latent_dims: int,
        encoder: BaseEncoder = Encoder,
        decoders=None,
        latent_dropout=0,
        **kwargs
    ):
        """

        :param latent_dims: # latent dimensions
        :param encoder: list of encoder networks
        :param decoders:  list of decoder networks
        """
        super().__init__(latent_dims=latent_dims, **kwargs)
        self.encoder = encoder
        self.decoders = torch.nn.ModuleList(decoders)

    def forward(self, *args, **kwargs):
        z = self.encoder(args[0])
        return z

    def _decode(self, z):
        """
        This method is used to decode from the latent space to the best prediction of the original views

        :param z:
        """
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(z))
        return tuple(recon)

    def loss(self, *args):
        z = self(*args)
        recons = self._decode(z)
        loss = dict()
        loss["reconstruction"] = torch.stack(
            [self.recon_loss(x, recon) for x, recon in zip(args, recons)]
        ).sum()
        loss["objective"] = loss["reconstruction"]
        return loss

    @staticmethod
    def recon_loss(x, recon):
        return F.mse_loss(recon, x, reduction="mean")
