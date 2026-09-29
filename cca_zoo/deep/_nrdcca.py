"""Deep CCA with noise regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep.objectives import MCCALoss, _inv_sqrtm


def _mean_canonical_correlation(
    a: torch.Tensor, b: torch.Tensor, eps: float
) -> torch.Tensor:
    """Mean canonical correlation of two batches, with ridge ``eps``."""
    a, b = a - a.mean(dim=0), b - b.mean(dim=0)
    n = a.shape[0]

    def whitener(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        return _inv_sqrtm(x.T @ x / (n - 1) + eps * eye, eps)

    t = whitener(a) @ (a.T @ b / (n - 1)) @ whitener(b)
    squared = torch.linalg.eigvalsh(t.T @ t).clamp(min=eps)
    return torch.sqrt(squared).mean()


class NRDCCA(BaseDeep):
    r"""Deep CCA with noise regularisation, against model collapse.

    Linear CCA's correlation of a view with independent noise does not change
    under an invertible linear map, $\operatorname{Corr}(X_k, A_k) =
    \operatorname{Corr}(W_k X_k, W_k A_k)$; an overfitted network's does. Each
    batch draws Gaussian noise $A_k$ of each view's shape, and the loss

    $$
    \mathcal{L} = \mathcal{L}_{DCCA}\bigl(f_1(X_1), \dots, f_K(X_K)\bigr)
        + \alpha \sum_{k=1}^{K} \bigl\lvert
        \operatorname{Corr}(f_k(X_k), f_k(A_k)) - \operatorname{Corr}(X_k, A_k)
        \bigr\rvert
    $$

    holds the encoders to that property. $\mathcal{L}_{DCCA}$ is
    :class:`~cca_zoo.deep.objectives.MCCALoss`, :class:`DCCA`'s loss summed
    over pairs of views, and $\operatorname{Corr}$ is the mean canonical
    correlation, which compares a view's $d_k$ canonical correlations with its
    encoding's. With ``alpha=0`` this is :class:`DMCCA`.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        alpha: Weight of the noise regularisation. Default is 1.0.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Ridge of the within-view covariances. Default is 1e-6.

    Raises:
        ValueError: If ``alpha`` is negative.

    References:
        He et al. (2024). Preventing Model Collapse in Deep Canonical
        Correlation Analysis by Noise Regularization. NeurIPS.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import NRDCCA
        >>> model = NRDCCA(n_components=4, encoders=[nn.Linear(10, 4), nn.Linear(8, 4)])
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        alpha: float = 1.0,
        learning_rate: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.alpha = alpha
        self.eps = eps
        self.objective = MCCALoss(eps=eps)

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The DCCA loss and the noise regularisation of a batch.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "dcca", "noise_regularisation"}``.
        """
        views = batch["views"]
        noise = [torch.randn_like(v) for v in views]
        encodings, noise_encodings = self(views), self(noise)
        regularisation = torch.stack(
            [
                torch.abs(
                    _mean_canonical_correlation(f, g, self.eps)
                    - _mean_canonical_correlation(x, a, self.eps)
                )
                for x, a, f, g in zip(views, noise, encodings, noise_encodings)
            ]
        ).sum()
        dcca = self.objective(encodings)
        return {
            "objective": dcca + self.alpha * regularisation,
            "dcca": dcca,
            "noise_regularisation": regularisation,
        }
