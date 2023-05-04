import torch

from .._base import _BaseDeep
from .._generative._base import _GenerativeMixin
from ..architectures import Encoder
from ..callbacks import GenerativeCallback


class SplitAE(_BaseDeep, _GenerativeMixin):
    """
    A class used to fit a Split Autoencoder model.

    References
    ----------
    Ngiam, Jiquan, et al. "Multimodal deep learning." ICML. 2011.

    """

    def __init__(
        self,
        latent_dims: int,
        encoder=Encoder,
        decoders=None,
        latent_dropout=0,
        recon_loss_type="mse",
        img_dim=None,
        **kwargs
    ):
        """

        :param latent_dims: # latent dimensions
        :param encoder: list of encoder networks
        :param decoders:  list of decoder networks
        """
        super().__init__(latent_dims=latent_dims, **kwargs)
        self.img_dim = img_dim
        self.encoder = encoder
        self.decoders = torch.nn.ModuleList(decoders)
        self.latent_dropout = torch.nn.Dropout(p=latent_dropout)
        self.recon_loss_type = recon_loss_type

    def forward(self, views, **kwargs):
        """
        Forward method for the model. Outputs latent encoding for each view

        :param views:
        :param kwargs:
        :return:
        """
        z = []
        z.append(self.encoder(views[0]))
        return z

    def _decode(self, z, **kwargs):
        """
        This method is used to decode from the latent space to the best prediction of the original views

        :param z:
        """
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(self.latent_dropout(z[0])))
        return recon

    def loss(self, views, **kwargs):
        z = self(views)
        recons = self._decode(z)
        loss = dict()
        loss["reconstruction"] = torch.stack(
            [
                self.recon_loss(x, recon, loss_type=self.recon_loss_type)
                for x, recon in zip(views, recons)
            ]
        ).sum()
        loss["objective"] = loss["reconstruction"]
        return loss

    def configure_callbacks(self):
        return [GenerativeCallback()]
