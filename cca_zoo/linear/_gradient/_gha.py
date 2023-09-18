import torch

from cca_zoo.linear._gradient._ey import CCA_EY


class CCA_GHA(CCA_EY):
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

    def loss(self, views, independent_views=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)
        rewards = torch.trace(2 * A)
        if independent_views is None:
            # Hebbian
            penalties = torch.trace(A.detach() @ B)
            # penalties = torch.trace(A @ B)
        else:
            # Encoding another set of views with the forward method
            independent_z = self(independent_views)
            # Getting A' and B' matrices from independent_z
            independent_A, independent_B = self.get_AB(independent_z)
            # Hebbian
            penalties = torch.trace(independent_A.detach() @ B)
            # penalties = torch.trace(A @ independent_B)
        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
