from typing import Union

import torch

from cca_zoo.models._plsmixin import PLSMixin
from cca_zoo.models._iterative._ey import CCAEY, EYLoop


class CCASVD(CCAEY):
    def _get_module(self, weights=None, k=None):
        return SVDLoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="cca",
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class PLSSVD(CCASVD, PLSMixin):
    def _get_module(self, weights=None, k=None):
        return SVDLoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="pls",
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class SVDLoop(EYLoop):
    def loss(self, views, views2=None, **kwargs):
        z = self(views)
        C = torch.cov(torch.hstack(z).T)
        latent_dims = z[0].shape[1]

        Cxy = C[:latent_dims, latent_dims:]
        Cxx = C[:latent_dims, :latent_dims]

        if views2 is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            z2 = self(views2)
            Cyy = torch.cov(torch.hstack(z2).T)[latent_dims:, latent_dims:]

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)

        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
