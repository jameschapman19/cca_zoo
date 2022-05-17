import torch

from cca_zoo.deepmodels._architectures import BaseEncoder, Encoder
from ._base import _BaseDeep, _GenerativeMixin


class SplitAE(_GenerativeMixin, _BaseDeep):
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
            recon_loss="mse",
            **kwargs
    ):
        """

        :param latent_dims: # latent dimensions
        :param encoder: list of encoder networks
        :param decoders:  list of decoder networks
        """
        super().__init__(latent_dims=latent_dims, recon_loss=recon_loss, **kwargs)
        self.encoder = encoder
        self.decoders = torch.nn.ModuleList(decoders)
        self.latent_dropout = torch.nn.Dropout(p=latent_dropout)

    def forward(self, views, **kwargs):
        z = self.encoder(views[0])
        return z

    def _decode(self, z):
        """
        This method is used to decode from the latent space to the best prediction of the original views

        :param z:
        """
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(self.latent_dropout(z)))
        return tuple(recon)

    def loss(self, views, **kwargs):
        z = self(views)
        recons = self._decode(z)
        loss = dict()
        loss["reconstruction"] = torch.stack(
            [self.recon_loss(x, recon) for x, recon in zip(views, recons)]
        ).sum()
        loss["objective"] = loss["reconstruction"]
        return loss
