import torch

from .._discriminative._dcca import DCCA
from .._discriminative._dmcca import _MCCALoss
from . import _GenerativeMixin


class DCCAE(DCCA, _GenerativeMixin):
    """
    A class used to fit a DCCAE model.

    References
    ----------
    Wang, Weiran, et al. "On deep multi-view representation learning." International conference on machine learning. PMLR, 2015.

    """

    objective = _MCCALoss()

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

    def _decode(self, z, **kwargs):
        """
        This method is used to decode from the latent space to the best prediction of the original representations

        """
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(self.latent_dropout(z[i])))
        return recon

    def loss(self, batch, **kwargs):
        z = self(batch["views"])
        recons = self._decode(z)
        loss = dict()
        loss["reconstruction"] = torch.stack(
            [
                self.recon_loss(x, recon, loss_type=self.recon_loss_type)
                for x, recon in zip(batch["views"], recons)
            ]
        ).sum()
        loss["correlation"] = self.objective(z)
        loss["objective"] = (
            self.lam * loss["reconstruction"] + (1 - self.lam) * loss["correlation"]
        )
        return loss
