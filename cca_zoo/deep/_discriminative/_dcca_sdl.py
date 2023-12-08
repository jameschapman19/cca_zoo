import torch
import torch.nn.functional as F

from ._dcca import DCCA


def sdl_loss(view):
    """Calculate SDL loss."""
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
        z = []
        for i, (encoder, bn) in enumerate(zip(self.encoders, self.bns)):
            z.append(bn(encoder(views[i])))
        return z

    def loss(self, batch, **kwargs):
        z = self(batch["views"])
        l2_loss = F.mse_loss(z[0], z[1])
        SDL_loss = torch.sum(torch.stack([sdl_loss(z_) for z_ in z]))
        loss = l2_loss + self.lam * SDL_loss
        return {"objective": loss, "l2": l2_loss, "sdl": SDL_loss}

    def _sdl_loss(self, covs):
        loss = 0
        for cov in covs:
            sgn = torch.sign(cov)
            sgn.fill_diagonal_(0)
            loss += torch.mean(cov * sgn)
        return loss
