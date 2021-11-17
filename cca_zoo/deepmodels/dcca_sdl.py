import torch
import torch.nn.functional as F

from cca_zoo.deepmodels import DCCA_NOI


class DCCA_SDL(DCCA_NOI):
    """
    A class used to fit a Deep CCA by Stochastic Decorrelation model.

    :Citation:

    Chang, Xiaobin, Tao Xiang, and Timothy M. Hospedales. "Scalable and effective deep CCA via soft decorrelation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

    """

    def __init__(
        self,
        latent_dims: int,
        N: int,
        encoders=None,
        r: float = 0,
        rho: float = 0.2,
        eps: float = 1e-3,
        shared_target: bool = False,
        lam=0.5,
    ):
        """
        Constructor class for DCCA
        :param latent_dims: # latent dimensions
        :param encoders: list of encoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA
        :param rho: covariance memory like DCCA non-linear orthogonal iterations paper
        :param eps: epsilon used throughout
        :param shared_target: not used
        """
        super().__init__(
            latent_dims=latent_dims,
            N=N,
            encoders=encoders,
            r=r,
            rho=rho,
            eps=eps,
            shared_target=shared_target,
        )
        self.c = None
        self.cross_cov = None
        self.lam = lam
        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(latent_dims, affine=False)
                for _ in range(latent_dims)
            ]
        )

    def forward(self, *args):
        z = []
        for i, (encoder, bn) in enumerate(zip(self.encoders, self.bns)):
            z.append(bn(encoder(args[i])))
        return tuple(z)

    def loss(self, *args):
        z = self(*args)
        self._update_covariances(*z, train=self.training)
        SDL_loss = self._sdl_loss(self.covs)
        l2_loss = F.mse_loss(z[0], z[1])
        return l2_loss + self.lam * SDL_loss

    def _sdl_loss(self, covs):
        loss = 0
        for cov in covs:
            cov = cov
            sgn = torch.sign(cov)
            sgn.fill_diagonal_(0)
            loss += torch.mean(cov * sgn)
        return loss

    def _update_covariances(self, *z, train=True):
        batch_covs = [z_.T @ z_ for z_ in z]
        if train:
            if self.c is not None:
                self.c = self.rho * self.c + 1
                self.covs = [
                    self.rho * self.covs[i].detach() + (1 - self.rho) * batch_cov
                    for i, batch_cov in enumerate(batch_covs)
                ]
            else:
                self.c = 1
                self.covs = batch_covs
        # pytorch-lightning runs validation once so this just fixes the bug
        elif self.covs is None:
            self.covs = batch_covs
