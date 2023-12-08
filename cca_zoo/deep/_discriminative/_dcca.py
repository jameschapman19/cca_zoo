from typing import List

import torch

from cca_zoo.deep._base import BaseDeep
from cca_zoo.deep._utils import inv_sqrtm
from cca_zoo.linear._mcca import MCCA


# Original work Copyright (c) 2016 Vahid Noroozi
# Modified work Copyright 2019 Zhanghao Wu

# Permission is hereby granted, free of chviewe, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:


# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
class _CCALoss:
    """Differentiable CCA Loss. Solves the CCA problem."""

    def correlation(self, representations: List[torch.Tensor]):
        """Calculate correlation."""
        latent_dims = representations[0].shape[1]
        o1 = representations[0].shape[1]
        o2 = representations[1].shape[1]

        representations = [
            representation - representation.mean(dim=0)
            for representation in representations
        ]

        SigmaHat12 = torch.cov(
            torch.hstack((representations[0], representations[1])).T
        )[:latent_dims, latent_dims:]
        SigmaHat11 = torch.cov(representations[0].T) + 1e-5 * torch.eye(
            o1, device=representations[0].device
        )
        SigmaHat22 = torch.cov(representations[1].T) + 1e-5 * torch.eye(
            o2, device=representations[1].device
        )

        SigmaHat11RootInv = inv_sqrtm(SigmaHat11, 1e-5)
        SigmaHat22RootInv = inv_sqrtm(SigmaHat22, 1e-5)

        Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
        trace_TT = Tval.T @ Tval
        eigvals = torch.linalg.eigvalsh(trace_TT)

        return eigvals

    def __call__(self, representations: List[torch.Tensor]):
        """Calculate loss."""
        eigvals = self.correlation(representations)
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        return -eigvals.sum()


class DCCA(BaseDeep):
    """
    A class used to fit a DCCA model.

    References
    ----------
    Andrew, Galen, et al. "Deep canonical correlation analysis." International conference on machine learning. PMLR, 2013.

    """

    objective = _CCALoss()

    def __init__(
        self,
        latent_dimensions: int,
        encoders=None,
        **kwargs,
    ):
        super().__init__(latent_dimensions=latent_dimensions, **kwargs)
        # Check if encoders are provided and have the same length as the number of representations
        if encoders is None:
            raise ValueError(
                "Encoders must be a list of torch.nn.Module with length equal to the number of representations."
            )
        self.encoders = torch.nn.ModuleList(encoders)

    def forward(self, views, **kwargs):
        if not hasattr(self, "n_views_"):
            self.n_views_ = len(views)
        # Use list comprehension to encode each view
        z = [encoder(view) for encoder, view in zip(self.encoders, views)]
        return z

    def loss(self, batch, **kwargs):
        representations = self(batch["views"])
        return {"objective": self.objective(representations)}

    def pairwise_correlations(self, loader: torch.utils.data.DataLoader):
        # Call the parent class method
        return super().pairwise_correlations(loader)

    def correlation_captured(self, z):
        # Remove mean from each view
        z = [zi - zi.mean(0) for zi in z]
        return MCCA(latent_dimensions=self.latent_dimensions).fit(z).score(z).sum()

    def score(self, loader: torch.utils.data.DataLoader, **kwargs):
        z = self.transform(loader)
        corr = self.correlation_captured(z)
        return corr
