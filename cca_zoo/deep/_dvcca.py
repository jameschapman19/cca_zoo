"""Deep variational CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep._dccae import _reconstruction_loss


class DVCCA(BaseDeep):
    r"""Deep variational CCA: a shared latent variable decoded to every view.

    Each view's encoder gives a Gaussian posterior
    $\mathcal{N}(\mu_i, \sigma_i^2)$; these are combined with the
    $\mathcal{N}(0, I)$ prior by a product of experts, which with one view is
    the original DVCCA posterior. Training minimises the negative ELBO

    $$
    \mathcal{L} = \sum_i \operatorname{MSE}(x_i, \text{decoder}_i(z))
        + \mathrm{KL}(q(z \mid x_1, \dots, x_M) \,\|\, \mathcal{N}(0, I)).
    $$

    Each view's encoding is its own posterior mean $\mu_i$.

    Args:
        n_components: Latent dimension.
        encoders: One module per view with ``2 * n_components`` outputs, the
            mean then the log-variance.
        decoders: One module per view mapping the latent back to that view.
        learning_rate: Adam learning rate. Default is 1e-3.

    References:
        Wang, W., Yan, X., Lee, H., & Livescu, K. (2016). Deep variational
        canonical correlation analysis. arXiv:1610.03454.

        Wu, M., & Goodman, N. (2018). Multimodal generative models for
        scalable weakly-supervised learning. NeurIPS.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DVCCA
        >>> model = DVCCA(
        ...     n_components=4,
        ...     encoders=[nn.Linear(10, 8), nn.Linear(10, 8)],
        ...     decoders=[nn.Linear(4, 10), nn.Linear(4, 10)],
        ... )
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        decoders: list[nn.Module],
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.decoders = nn.ModuleList(decoders)

    def _posteriors(
        self, views: list[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``(mu, log_var)`` of each view's posterior."""
        k = self.n_components
        posteriors = []
        for enc, v in zip(self.encoders, views):
            out = enc(v)
            if out.shape[1] != 2 * k:
                raise ValueError(
                    f"A DVCCA encoder returned {out.shape[1]} outputs; it needs "
                    f"2 * n_components = {2 * k}."
                )
            posteriors.append((out[:, :k], out[:, k:]))
        return posteriors

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Each view's posterior mean, shape (batch_size, n_components)."""
        return [mu for mu, _ in self._posteriors(views)]

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The negative ELBO of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "reconstruction", "kl"}``.
        """
        views = batch["views"]
        posteriors = self._posteriors(views)
        means = torch.stack([m for m, _ in posteriors])
        precisions = torch.stack([torch.exp(-log_var) for _, log_var in posteriors])
        var = 1.0 / (1.0 + precisions.sum(dim=0))
        mu = (means * precisions).sum(dim=0) * var
        z = mu + torch.randn_like(mu) * var.sqrt()
        reconstruction = _reconstruction_loss(views, [dec(z) for dec in self.decoders])
        kl = 0.5 * torch.sum(var + mu.pow(2) - 1.0 - var.log()) / mu.shape[0]
        return {
            "objective": reconstruction + kl,
            "reconstruction": reconstruction,
            "kl": kl,
        }
