import torch

from cca_zoo.linear._gradient._ey import CCA_EY


class CCA_SVD(CCA_EY):
    def _more_tags(self):
        return {"multiview": False, "stochastic": True}

    def loss(self, views, independent_views=None, **kwargs):
        z = self(views)
        C = torch.cov(torch.hstack(z).T)
        latent_dims = z[0].shape[1]

        Cxy = (C[:latent_dims, latent_dims:] + C[latent_dims:, :latent_dims]) / 2
        Cxx = C[:latent_dims, :latent_dims]

        if independent_views is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            independent_z = self(independent_views)
            Cyy = torch.cov(torch.hstack(independent_z).T)[latent_dims:, latent_dims:]

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)

        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }


class PLS_SVD(CCA_SVD):
    def loss(self, views, independent_views=None, **kwargs):
        z = self(views)
        C = torch.cov(torch.hstack(z).T)
        latent_dims = z[0].shape[1]

        n = z[0].shape[0]
        Cxy = C[:latent_dims, latent_dims:]
        Cxx = self.torch_weights[0].T @ self.torch_weights[0] / n
        Cyy = self.torch_weights[1].T @ self.torch_weights[1] / n

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)

        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
