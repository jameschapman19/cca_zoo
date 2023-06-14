from typing import Iterable

import torch
from torchmetrics import Metric

from cca_zoo.deep.objectives import _demean, inv_sqrtm


class MCCA(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("representations", default=[], persistent=False)

    def update(self, representations: Iterable[torch.Tensor] = None):
        for i, representation in enumerate(representations):
            if len(self.representations) < len(representations):
                self.representations.append([representation])
            else:
                self.representations[i].append(representation)
        self.latent_dims = representations[0].shape[1]

    @torch.no_grad()
    def compute(self):
        self.representations = [
            torch.cat(representation) for representation in self.representations
        ]
        return self.correlation(self.representations)

    def C(self, views):
        # Concatenate all views and from this get the cross-covariance matrix
        all_views = torch.cat(views, dim=1)
        C = torch.cov(all_views.T)
        C = C - torch.block_diag(*[torch.cov(view.T) for view in views])
        return C

    def D(self, views):
        # Get the block covariance matrix placing Xi^TX_i on the diagonal
        D = torch.block_diag(*[torch.cov(view.T) for i, view in enumerate(views)])
        return D

    def correlation(self, views):
        # Subtract the mean from each output
        views = _demean(views)

        C = self.C(views)
        D = self.D(views)

        R = inv_sqrtm(D, eps=1e-9)

        # In MCCA our eigenvalue problem Cv = lambda Dv
        C_whitened = R @ C @ R.T

        eigvals = torch.linalg.eigvalsh(C_whitened)

        # Sort eigenvalues so lowest first
        idx = torch.argsort(eigvals, descending=True)

        eigvals = eigvals[idx[: self.latent_dims]]

        return eigvals / 2
