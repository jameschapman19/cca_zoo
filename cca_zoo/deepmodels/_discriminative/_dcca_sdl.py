import torch
import torch.nn.functional as F

from ._dcca_noi import DCCA_NOI


class DCCA_SDL(DCCA_NOI):
    """
    A class used to fit a Deep CCA by Stochastic Decorrelation model.

    References
    ----------
    Chang, Xiaobin, Tao Xiang, and Timothy M. Hospedales. "Scalable and effective deep CCA via soft decorrelation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

    """

    def __init__(
        self,
        latent_dims: int,
        N: int,
        encoders=None,
        r: float = 0,
        rho: float = 0.2,
        eps: float = 1e-5,
        shared_target: bool = False,
        lam=0.5,
        **kwargs
    ):
        super().__init__(
            latent_dims=latent_dims,
            N=N,
            encoders=encoders,
            r=r,
            rho=rho,
            eps=eps,
            shared_target=shared_target,
            **kwargs
        )
        self.c = None
        self.cross_cov = None
        self.lam = lam
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(latent_dims, affine=False) for _ in self.encoders]
        )

    def forward(self, views, **kwargs):
        z = []
        for i, (encoder, bn) in enumerate(zip(self.encoders, self.bns)):
            z.append(bn(encoder(views[i])))
        return z

    def loss(self, views, **kwargs):
        z = self(views)
        l2_loss = F.mse_loss(z[0], z[1])
        self._update_covariances(z, train=self.training)
        SDL_loss = self._sdl_loss(self.covs)
        loss = l2_loss + self.lam * SDL_loss
        self.covs = [cov.detach() for cov in self.covs]
        return {"objective": loss, "l2": l2_loss, "sdl": SDL_loss}

    def _sdl_loss(self, covs):
        loss = 0
        for cov in covs:
            sgn = torch.sign(cov)
            sgn.fill_diagonal_(0)
            loss += torch.mean(cov * sgn)
        return loss

    def _update_covariances(self, z, train=True):
        b = z[0].shape[0]
        batch_covs = [self.N * z_.T @ z_ / b for z_ in z]
        if train:
            if self.covs is not None:
                self.covs = [
                    self.rho * self.covs[i] + (1 - self.rho) * batch_cov
                    for i, batch_cov in enumerate(batch_covs)
                ]
            else:
                self.covs = batch_covs
        # pytorch-lightning runs validation once so this just fixes the bug
        elif self.covs is None:
            self.covs = batch_covs
