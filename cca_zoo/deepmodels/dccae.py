import torch
from torch.nn import functional as F

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.architectures import Encoder, Decoder
from cca_zoo.deepmodels.dcca import _DCCA_base


class DCCAE(_DCCA_base):
    """
    A class used to fit a DCCAE model.

    :Citation:

    Wang, Weiran, et al. "On deep multi-view representation learning." International conference on machine learning. PMLR, 2015.

    """

    def __init__(
        self,
        latent_dims: int,
        objective=objectives.MCCA,
        encoders=None,
        decoders=None,
        r: float = 0,
        eps: float = 1e-3,
        lam=0.5,
    ):
        """
        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param decoders:  list of decoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        :param lam: weight of reconstruction loss (1 minus weight of correlation loss)
        """
        super().__init__(latent_dims=latent_dims)
        if decoders is None:
            decoders = [Decoder, Decoder]
        if encoders is None:
            encoders = [Encoder, Encoder]
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        if lam < 0 or lam > 1:
            raise ValueError(f"lam should be between 0 and 1. rho={lam}")
        self.lam = lam
        self.objective = objective(latent_dims, r=r, eps=eps)

    def forward(self, *args):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return z

    def recon(self, *args):
        """
        :param args:
        :return:
        """
        z = self(*args)
        return self._decode(*z)

    def _decode(self, *z):
        """
        This method is used to decode from the latent space to the best prediction of the original views

        """
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(z[i]))
        return recon

    def loss(self, *args):
        z = self(*args)
        recon = self._decode(*z)
        recon_loss = self._recon_loss(args[: len(recon)], recon)
        return self.lam * recon_loss + self.objective.loss(*z)

    @staticmethod
    def _recon_loss(x, recon):
        recons = [
            F.binary_cross_entropy(recon_, x_, reduction="mean") for recon_, x_ in zip(recon, x)
        ]
        return torch.stack(recons).sum(dim=0)
