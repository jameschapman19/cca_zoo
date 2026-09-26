"""Deep CCA by nonlinear orthogonal iterations."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep.objectives import _inv_sqrtm


class _BatchWhiten(nn.Module):
    """Whitening layer with a running covariance; the identity in eval mode.

    Args:
        num_features: Input dimension.
        momentum: Running-covariance update rate. Default is 0.1.
        eps: Floor on the covariance eigenvalues. Default is 1e-5.
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
        """Whiten a batch of shape (batch_size, num_features)."""
        if not self.training:
            return x

        self.num_batches_tracked.add_(1)
        factor = self.momentum

        batch_cov = (x.T @ x) / x.shape[0]
        with torch.no_grad():
            self.running_covar.mul_(1.0 - factor).add_(batch_cov * factor)
            w = _inv_sqrtm(self.running_covar, self.eps)

        return x @ w


class DCCANOI(DCCA):
    r"""Deep CCA by nonlinear orthogonal iterations.

    Regresses each view's encoding on the others' whitened encodings, held
    fixed:

    $$
    \mathcal{L} = \sum_{i \neq j} \bigl\| z_i - \operatorname{sg}(W_j z_j) \bigr\|_2^2,
    $$

    with $W_j$ a running batch-whitening transform and sg a stop-gradient.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        rho: Running-covariance update rate in ``[0, 1]``. Default is 0.1.
        learning_rate: Adam learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Floor on the whitening eigenvalues. Default is 1e-6.

    Raises:
        ValueError: If ``rho`` is outside ``[0, 1]``.

    References:
        Wang, W., Arora, R., Livescu, K., & Srebro, N. (2015). Stochastic
        optimization for deep CCA via nonlinear orthogonal iterations.
        Allerton.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DCCANOI
        >>> encoders = [nn.Linear(10, 4), nn.Linear(8, 4)]
        >>> model = DCCANOI(n_components=4, encoders=encoders)
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        rho: float = 0.1,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.rho = rho
        self.mse = nn.MSELoss(reduction="sum")
        self.bws = nn.ModuleList(
            [_BatchWhiten(n_components, momentum=rho, eps=eps) for _ in encoders]
        )

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """The NOI loss of a batch.

        Args:
            representations: One encoded tensor per view.
            independent_representations: Unused.

        Returns:
            ``{"objective": loss}``.
        """
        whitened = [bw(r) for r, bw in zip(representations, self.bws)]
        total = torch.tensor(0.0, device=representations[0].device)
        n_views = len(representations)
        for i in range(n_views):
            for j in range(n_views):
                if i != j:
                    total = total + self.mse(representations[i], whitened[j].detach())
        return {"objective": total}
