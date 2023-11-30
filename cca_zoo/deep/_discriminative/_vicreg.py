import torch
import torch.nn.functional as F

from ._dcca import DCCA


class VICReg(DCCA):
    """
    A class used to fit a VICReg model.

    VICReg is a self-supervised learning method that applies redundancy-reduction
    to learn representations that are invariant to distortions of the input sample.

    Parameters
    ----------

    sim_loss_weight : float, optional
        weight of the similarity loss. Defaults to 25.0.
    var_loss_weight : float, optional
        weight of the variance loss. Defaults to 25.0.
    cov_loss_weight : float, optional
        weight of the covariance loss. Defaults to 1.0.

    References
    ----------
    .. [1] Bardes, A., Ponce, J. and LeCun, Y., 2021. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906.

    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders=None,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, **kwargs
        )
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    def loss(self, batch, **kwargs):
        z = self(batch["views"])  # get the latent representations
        sim_loss = invariance_loss(*z)
        var_loss = variance_loss(*z)
        cov_loss = covariance_loss(*z)
        loss = (
            self.sim_loss_weight * sim_loss
            + self.var_loss_weight * var_loss
            + self.cov_loss_weight * cov_loss
        )
        return {
            "objective": loss,  # return the loss
            "sim_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = (
        cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    )
    return cov_loss
