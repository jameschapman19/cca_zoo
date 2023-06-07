import torch

from cca_zoo.models._iterative._ey import CCAEY, EYLoop


class CCAGH(CCAEY):
    def _get_module(self, weights=None, k=None):
        return GHALoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="cca",
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class GHALoop(EYLoop):
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
                    B += torch.cov(zi.T)  # add the auto-covariance of each view to B
                A += torch.cov(torch.hstack((zi, zj)).T)[
                    latent_dims:, :latent_dims
                ]  # add the cross-covariance of each pair of views to A
        return A / len(z), B / len(
            z
        )  # return the normalized matrices (divided by the number of views)

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
            _, B_ = self.get_AB(z2)
            # Computing rewards and penalties using A and B'
            rewards = torch.trace(2 * A)
            penalties = torch.trace(A @ B_)
        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
