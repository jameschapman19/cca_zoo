from typing import List

import torch
import torch.nn.functional as F

from ._dcca import DCCA


def sdl_loss(view):
    """Calculate SDL minibatch_loss."""
    cov = torch.cov(view.T)
    sgn = torch.sign(cov)
    sgn.fill_diagonal_(0)
    return torch.mean(cov * sgn)


class DCCA_SDL(DCCA):
    """
    A class used to fit a Deep _CCALoss by Stochastic Decorrelation model.

    References
    ----------
    Chang, Xiaobin, Tao Xiang, and Timothy M. Hospedales. "Scalable and effective deep _CCALoss via soft decorrelation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

    """

    def __init__(self, *args, lam=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam
        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(self.latent_dimensions, affine=False)
                for _ in self.encoders
            ]
        )

    def forward(self, views, **kwargs):
        representations = []
        for i, (encoder, bn) in enumerate(zip(self.encoders, self.bns)):
            representations.append(bn(encoder(views[i])))
        return representations

    def loss(
        self,
        representations: List[torch.Tensor],
        independent_representations: List[torch.Tensor]=None,
    ):
        l2_loss = F.mse_loss(representations[0], representations[1])
        SDL_loss = torch.sum(
            torch.stack(
                [sdl_loss(representation) for representation in representations]
            )
        )
        loss = l2_loss + self.lam * SDL_loss
        return {"objective": loss, "l2": l2_loss, "sdl": SDL_loss}
