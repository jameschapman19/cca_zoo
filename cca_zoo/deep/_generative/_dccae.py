from typing import List

import torch

from . import _GenerativeMixin
from .. import DMCCA


class DCCAE(DMCCA, _GenerativeMixin):
    """
    A class used to fit a DCCAE model.

    References
    ----------
    Wang, Weiran, et al. "On deep multi-view representation learning." International conference on machine learning. PMLR, 2015.

    """

    def __init__(
        self,
        *args,
        decoders=None,
        lam=0.5,
        latent_dropout=0,
        img_dim=None,
        recon_loss_type="mse",
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.img_dim = img_dim
        self.decoders = torch.nn.ModuleList(decoders)
        if lam < 0 or lam > 1:
            raise ValueError(f"lam should be between 0 and 1. rho={lam}")
        self.lam = lam
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
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(views[i]))
        return z

    def _decode(self, representations, **kwargs):
        """
        This method is used to decode from the latent space to the best prediction of the original representations

        """
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(self.latent_dropout(representations[i])))
        return recon

    def minibatch_loss(self, batch, **kwargs):
        # Encoding the representations with the forward method
        representations = self(batch["views"])
        if batch.get("independent_views") is None:
            independent_representations = None
        else:
            independent_representations = self(batch["independent_views"])
        recons = self._decode(representations)
        loss = DMCCA.loss(self, representations, independent_representations)
        loss["reconstruction"] = torch.stack(
            [
                self.recon_loss(x, recon, loss_type=self.recon_loss_type)
                for x, recon in zip(batch["views"], recons)
            ]
        ).sum()
        loss["objective"] = (
            self.lam * loss["reconstruction"] + (1 - self.lam) * loss["objective"]
        )
        return loss
