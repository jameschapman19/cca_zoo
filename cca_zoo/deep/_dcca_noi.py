"""DCCA_NOI — Deep CCA via Non-linear Orthogonal Iterations (Wang 2015)."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep.objectives import _inv_sqrtm


class _BatchWhiten(nn.Module):
    """Batch whitening layer with exponential moving average covariance.

    Tracks a running estimate of the feature covariance and whitens
    the input using its inverse square root.  Only applied during
    training; at eval time the input is returned unchanged.

    Args:
        num_features: Dimensionality of the input features.
        momentum: Exponential moving average factor for the running
            covariance. Default is 0.1.
        eps: Regularisation added to eigenvalues. Default is 1e-5.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer(
            "running_covar",
            torch.eye(num_features),
        )
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long),
        )
        self.running_covar: torch.Tensor
        self.num_batches_tracked: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten the input tensor.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Whitened tensor of the same shape.
        """
        if not self.training:
            return x

        self.num_batches_tracked.add_(1)
        factor = self.momentum

        batch_cov = (x.T @ x) / x.shape[0]
        with torch.no_grad():
            self.running_covar.mul_(1.0 - factor).add_(batch_cov * factor)
            w = _inv_sqrtm(self.running_covar, self.eps)

        return x @ w


class DCCA_NOI(DCCA):
    r"""Deep CCA via Non-linear Orthogonal Iterations.

    Uses batch whitening to approximate the CCA whitening step
    stochastically. The loss pushes each view's representations towards a
    stop-gradient copy of the other views' whitened representations:

    $$
    \mathcal{L} = \sum_{i \neq j}
        \left\| z_i - \operatorname{sg}\!\bigl(W_j z_j\bigr) \right\|_2^2
    $$

    where $W_j$ is an exponential-moving-average batch-whitening
    transform for view $j$ (see :class:`_BatchWhiten`) and
    $\operatorname{sg}(\cdot)$ denotes stop-gradient.

    References:
        Wang, W., et al. "Stochastic optimization for deep CCA via
        nonlinear orthogonal iterations." Allerton 2015. IEEE.

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        rho: Exponential moving average momentum for the batch whitening
            layers. Must be in [0, 1]. Default is 0.1.
        objective: Ignored; the NOI loss is fixed. Accepted for API
            compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Regularisation for the whitening layers. Default is 1e-6.

    Raises:
        ValueError: If ``rho`` is not in [0, 1].

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = DCCA_NOI(latent_dimensions=4, encoders=[enc1, enc2], rho=0.1)
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        rho: float = 0.1,
        objective: nn.Module | None = None,
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}.")
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            objective=objective,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.rho = rho
        self.mse = nn.MSELoss(reduction="sum")
        self.bws = nn.ModuleList(
            [_BatchWhiten(latent_dimensions, momentum=rho, eps=eps) for _ in encoders]
        )

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the NOI loss.

        Each view's representations are pushed towards the whitened
        representation of the other view (stop-gradient on the target).

        Args:
            representations: Encoded views from the current batch, each
                of shape (batch_size, latent_dimensions).
            independent_representations: Unused; present for API
                compatibility.

        Returns:
            Dictionary with key ``"objective"``.
        """
        whitened = [bw(r) for r, bw in zip(representations, self.bws)]
        total = torch.tensor(0.0, device=representations[0].device)
        n_views = len(representations)
        for i in range(n_views):
            for j in range(n_views):
                if i != j:
                    total = total + self.mse(representations[i], whitened[j].detach())
        return {"objective": total}
