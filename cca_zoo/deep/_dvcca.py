"""Deep variational CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep._dccae import _reconstruction_loss


class DVCCA(BaseDeep):
    r"""Deep variational CCA: a latent inferred from one view generates all of them.

    With prior $z \sim \mathcal{N}(0, I)$, decoders $p(x_i \mid z)$ for every
    view and an encoder $q(z \mid x_1) = \mathcal{N}(\mu, \sigma^2)$ of the
    first view, training minimises the negative ELBO

    $$
    \mathcal{L} = \sum_i \operatorname{MSE}(x_i, \text{decoder}_i(z))
        + \mathrm{KL}(q(z \mid x_1) \,\|\, \mathcal{N}(0, I)).
    $$

    Unlike the other deep models there is one encoding, so ``trainer.predict``
    returns a single array, the posterior mean $\mu$, with no linear CCA.

    Args:
        n_components: Latent dimension.
        encoder: Module mapping the first view to ``2 * n_components`` outputs,
            the mean then the log-variance.
        decoders: One module per view mapping the latent back to that view.
        learning_rate: Adam learning rate. Default is 1e-3.

    References:
        Wang, W., Yan, X., Lee, H., & Livescu, K. (2016). Deep variational
        canonical correlation analysis. arXiv:1610.03454.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DVCCA
        >>> model = DVCCA(
        ...     n_components=4,
        ...     encoder=nn.Linear(10, 8),
        ...     decoders=[nn.Linear(4, 10), nn.Linear(4, 6)],
        ... )
    """

    def __init__(
        self,
        n_components: int,
        encoder: nn.Module,
        decoders: list[nn.Module],
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=[encoder],
            learning_rate=learning_rate,
        )
        self.decoders = nn.ModuleList(decoders)

    def _posterior(
        self, views: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log-variance of ``q(z | x_1)``."""
        k = self.n_components
        out = self.encoders[0](views[0])
        if out.shape[1] != 2 * k:
            raise ValueError(
                f"The DVCCA encoder returned {out.shape[1]} outputs; it needs "
                f"2 * n_components = {2 * k}."
            )
        return out[:, :k], out[:, k:]

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """The posterior mean from the first view, as a one-element list."""
        return [self._posterior(views)[0]]

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The negative ELBO of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "reconstruction", "kl"}``.
        """
        views = batch["views"]
        mu, log_var = self._posterior(views)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        reconstruction = _reconstruction_loss(views, [dec(z) for dec in self.decoders])
        kl = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]
        return {
            "objective": reconstruction + kl,
            "reconstruction": reconstruction,
            "kl": kl,
        }

    def predict_step(self, batch: Batch, batch_idx: int) -> list[torch.Tensor]:
        """The posterior mean of a batch, as a one-element list."""
        return self(batch["views"])

    def on_train_end(self) -> None:
        """Nothing to fit: the posterior mean is the output."""
