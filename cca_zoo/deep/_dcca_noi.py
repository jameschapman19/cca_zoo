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
        rho: Weight of the previous running moments in ``[0, 1)``, the
            paper's time constant. Default is 0.9.
        reg_covar: Regularisation added to the running covariance's diagonal.
            Default is 1e-6.
    """

    def __init__(
        self,
        num_features: int,
        rho: float = 0.9,
        reg_covar: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.rho = rho
        self.reg_covar = reg_covar
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
                self.running_mean.lerp_(x.mean(dim=0), 1.0 - self.rho)
                self.running_covar.lerp_(
                    centred.T @ centred / (x.shape[0] - 1), 1.0 - self.rho
                )
                self.num_batches_tracked.add_(1)
        eye = torch.eye(self.num_features, device=x.device, dtype=x.dtype)
        whitener = _inv_sqrtm(self.running_covar + self.reg_covar * eye, self.reg_covar)
        return (x - self.running_mean) @ whitener


class DCCANOI(BaseDeep):
    r"""Deep CCA by nonlinear orthogonal iterations.

    Wang et al.'s Algorithm 2: regresses each view's encoding on the others'
    whitened, centred encodings, held fixed,

    $$
    \mathcal{L} = \sum_{i \neq j} \bigl\| z_i
        - \operatorname{sg}\bigl(\Sigma_j^{-1/2} (z_j - \mu_j)\bigr) \bigr\|_2^2,
    $$

    with sg a stop-gradient, and the running moments updated by each training
    batch as $\Sigma_j \leftarrow \rho \Sigma_j + (1 - \rho)\,
    \widehat{\Sigma}_j$, and $\mu_j$ likewise.

    Args:
        n_components: Number of latent dimensions.
        encoders: One module per view.
        rho: Weight of the previous running moments in ``[0, 1)``, the paper's
            time constant. Default is 0.9.
        learning_rate: Adam learning rate. Default is 1e-3.
        reg_covar: Non-negative regularisation added to the diagonal of each
            covariance, as scikit-learn's ``GaussianMixture``. Default is 1e-6.

    Raises:
        ValueError: If ``rho`` is outside ``[0, 1)``.

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
        rho: float = 0.9,
        learning_rate: float = 1e-3,
        reg_covar: float = 1e-6,
    ) -> None:
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.reg_covar = reg_covar
        self.rho = rho
        self.mse = nn.MSELoss(reduction="sum")
        self.bws = nn.ModuleList(
            [_BatchWhiten(n_components, rho=rho, reg_covar=reg_covar) for _ in encoders]
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
