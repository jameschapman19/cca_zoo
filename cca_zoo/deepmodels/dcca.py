import torch

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.architectures import Encoder
from cca_zoo.models import MCCA
from ._dcca_base import _DCCA_base


class DCCA(_DCCA_base):
    """
    A class used to fit a DCCA model.

    Examples
    --------
    >>> from cca_zoo.deepmodels import DCCA
    >>> model = DCCA()
    """

    def __init__(self, latent_dims: int, objective=objectives.MCCA,
                 encoders=None,
                 r: float = 0, eps: float = 1e-3):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        """
        super().__init__(latent_dims=latent_dims)
        if encoders is None:
            encoders = [Encoder, Encoder]
        self.encoders = torch.nn.ModuleList(encoders)
        self.objective = objective(latent_dims, r=r, eps=eps)

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return tuple(z)

    def loss(self, *args):
        z = self(*args)
        return self.objective.loss(*z)

    def post_transform(self, *z_list, train=False):
        if train:
            self.cca = MCCA(latent_dims=self.latent_dims)
            z_list = self.cca.fit_transform(*z_list)
        else:
            z_list = self.cca.transform(*z_list)
        return z_list
