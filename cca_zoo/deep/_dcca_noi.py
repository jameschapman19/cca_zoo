"""Deep CCA by nonlinear orthogonal iterations."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep.objectives import _inv_sqrtm


class _BatchWhiten(nn.Module):
    """Whitening layer by a running mean and covariance, updated in training mode.

    As :class:`~torch.nn.BatchNorm1d` keeps a running mean, so that an
    encoder's bias does not leak into the whitened target.

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
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer(
            "running_covar",
            torch.eye(num_features),
        )
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long),
        )
        self.running_mean: torch.Tensor
        self.running_covar: torch.Tensor
        self.num_batches_tracked: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten a batch of shape (batch_size, num_features) by the running moments.

        In training mode the batch first updates the running mean and covariance.
        """
        if self.training:
            with torch.no_grad():
                centred = x - x.mean(dim=0)
                self.running_mean.lerp_(x.mean(dim=0), self.momentum)
                self.running_covar.lerp_(
                    centred.T @ centred / (x.shape[0] - 1), self.momentum
                )
                self.num_batches_tracked.add_(1)
        return (x - self.running_mean) @ _inv_sqrtm(self.running_covar, self.eps)


class DCCANOI(BaseDeep):
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
        eps: float = 1e-6,
    ) -> None:
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.eps = eps
        self.rho = rho
        self.mse = nn.MSELoss(reduction="sum")
        self.bws = nn.ModuleList(
            [_BatchWhiten(n_components, momentum=rho, eps=eps) for _ in encoders]
        )

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The NOI loss of a batch.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective": loss}``.
        """
        representations = self(batch["views"])
        targets = [bw(z).detach() for z, bw in zip(representations, self.bws)]
        objective = sum(
            self.mse(representations[i], targets[j])
            for i, j in itertools.permutations(range(len(representations)), 2)
        )
        return {"objective": objective}
