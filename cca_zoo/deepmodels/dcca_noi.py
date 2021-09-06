from typing import Iterable

import numpy as np
import torch

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.dcca import DCCA
from cca_zoo.models import MCCA


class DCCA_NOI(DCCA):
    """
    A class used to fit a DCCA model by non-linear orthogonal iterations

    """

    def __init__(self, latent_dims: int, N: int,
                 encoders=None,
                 r: float = 0, rho: float = 0.2, eps: float = 1e-3, shared_target: bool = False):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param N: # samples used to estimate covariance
        :param encoders: list of encoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA
        :param rho: covariance memory like DCCA non-linear orthogonal iterations paper
        :param eps: epsilon used throughout
        :param shared_target: not used
        """
        super().__init__(latent_dims=latent_dims, encoders=encoders,
                         r=r, eps=eps)
        self.N = N
        self.covs = None
        if rho < 0 or rho > 1:
            raise ValueError(f"rho should be between 0 and 1. rho={rho}")
        self.eps = eps
        self.rho = rho
        self.shared_target = shared_target
        self.mse = torch.nn.MSELoss()
        # Authors state that a final linear layer is an important part of their algorithmic implementation
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(latent_dims, latent_dims, bias=False) for _ in range(len(encoders))])
        self.rand = torch.rand(N, self.latent_dims)

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = []
        #Users architecture + final linear layer
        for i, (encoder, linear_layer) in enumerate(zip(self.encoders, self.linear_layers)):
            z.append(linear_layer(encoder(args[i])))
        return tuple(z)

    def loss(self, *args):
        z = self(*args)
        z_copy = [z_.detach().clone() for z_ in z]
        self.update_covariances(*z_copy)
        covariance_inv = [torch.linalg.inv(objectives.MatrixSquareRoot.apply(cov)) for
                          cov in self.covs]
        preds = [z_ @ covariance_inv[i] for i, z_ in enumerate(z_copy)]
        loss = self.mse(z[0], preds[1]) + self.mse(z[1], preds[0])
        return loss

    def update_mean(self, *z):
        batch_means = [torch.mean(z_, dim=0) for z_ in z]
        if self.means is not None:
            self.means = [self.rho * self.means[i].detach() + (1 - self.rho) * batch_mean for i, batch_mean in
                          enumerate(batch_means)]
        else:
            self.means = batch_means
        z = [z_ - mean for (z_, mean) in zip(z, self.means)]
        return z

    def update_covariances(self, *z):
        b = z[0].shape[0]
        batch_covs = [self.N * z_.T @ z_ / b for z_ in z]
        if self.covs is not None:
            self.covs = [self.rho * self.covs[i] + (1 - self.rho) * batch_cov for i, batch_cov in enumerate(batch_covs)]
        else:
            self.covs = batch_covs

    def post_transform(self, *z_list, train=False) -> Iterable[np.ndarray]:
        if train:
            self.cca = MCCA(latent_dims=self.latent_dims)
            self.cca.fit(*z_list)
            z_list = self.cca.transform(*z_list)
        else:
            z_list = self.cca.transform(*z_list)
        return z_list
