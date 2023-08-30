import torch

from cca_zoo.linear._gradient._ey import CCAEY


class CCAGH(CCAEY):
    def _more_tags(self):
        return {"multiview": True, "stochastic": True}

    def get_AB(self, z):
        latent_dims = z[0].shape[1]
        A = torch.zeros(
            latent_dims, latent_dims, device=z[0].device
        )  # initialize the cross-covariance matrix
        B = torch.zeros(
            latent_dims, latent_dims, device=z[0].device
        )  # initialize the auto-covariance matrix
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    B += self._cross_covariance(zi, zj, latent_dims)
                A += self._cross_covariance(zi, zj, latent_dims)
        return A / len(z), B / len(z)

    def loss(self, views, views2=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)
        if views2 is None:
            # Computing rewards and penalties using A and B only
            rewards = torch.trace(2 * A)
            penalties = torch.trace(A @ B)
        else:
            # Encoding another set of views with the forward method
            z2 = self(views2)
            # Getting A' and B' matrices from z2
            A_, B_ = self.get_AB(z2)
            # Computing rewards and penalties using A and B'
            rewards = torch.trace(2 * A)
            penalties = torch.trace(A_ @ B)
        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
