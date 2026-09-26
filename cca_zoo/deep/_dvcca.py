"""Deep variational CCA."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import BaseDeep


class DVCCA(BaseDeep):
    r"""Deep variational CCA: a multiview variational autoencoder.

    Each encoder outputs a mean and log-variance, summed over views into
    $q(z \mid X)$; one decoder per view reconstructs it from $z$. Training
    minimises the negative ELBO

    $$
    \mathcal{L} = \sum_i \operatorname{MSE}(x_i, \text{decoder}_i(z))
        + \mathrm{KL}(q(z \mid X) \,\|\, \mathcal{N}(0, I)).
    $$

    Args:
        n_components: Latent dimension.
        encoders: One module per view mapping it to ``2 * n_components``
            outputs, the mean then the log-variance.
        decoders: One module per view mapping the latent back to that view.
        learning_rate: Adam learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.

    References:
        Wang, W., Yan, X., Lee, H., & Livescu, K. (2016). Deep variational
        canonical correlation analysis. arXiv:1610.03454.

    Example:
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
        max_epochs: int = 100,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
        )
        self.decoders = nn.ModuleList(decoders)

    def _encode(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log-variance of the shared posterior, summed over views."""
        k = self.n_components
        mu_sum = torch.zeros(views[0].shape[0], k, device=views[0].device)
        lv_sum = torch.zeros(views[0].shape[0], k, device=views[0].device)
        for enc, v in zip(self.encoders, views):
            out = enc(v)
            mu_sum = mu_sum + out[:, :k]
            lv_sum = lv_sum + out[:, k:]
        return mu_sum, lv_sum

    def _reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample the latent by the reparameterisation trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Reconstruct every view from the latent."""
        return [dec(z) for dec in self.decoders]

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """The shared posterior mean, as a one-element list.

        Args:
            views: One tensor per view.

        Returns:
            ``[mu]``, with ``mu`` of shape (batch_size, n_components).
        """
        mu, _ = self._encode(views)
        return [mu]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Zero objective; the ELBO needs the inputs, so it is computed in training.

        Args:
            representations: Unused.
            independent_representations: Unused.

        Returns:
            ``{"objective": 0}``.
        """
        return {"objective": torch.tensor(0.0)}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Negative ELBO of a batch.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.
            batch_idx: Unused.

        Returns:
            The loss.
        """
        views = batch["views"]
        mu, log_var = self._encode(views)
        z = self._reparameterise(mu, log_var)
        reconstructions = self._decode(z)

        recon_loss = torch.stack(
            [F.mse_loss(x, r) for x, r in zip(views, reconstructions)]
        ).sum()
        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp())
        n = views[0].shape[0]
        kl = kl / n

        objective = recon_loss + kl
        loss_dict: dict[str, torch.Tensor] = {
            "objective": objective,
            "reconstruction": recon_loss,
            "kl": kl,
        }
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=views[0].shape[0],
            )
        return objective

    @torch.no_grad()
    def transform(self, loader: torch.utils.data.DataLoader) -> list[np.ndarray]:
        """Posterior mean of the shared latent for every sample.

        Args:
            loader: DataLoader yielding batches with a ``"views"`` key.

        Returns:
            A one-element list holding an array of shape (n_samples, n_components).
        """
        self.eval()
        all_mu: list[torch.Tensor] = []
        for batch in loader:
            views_dev = [v.to(self.device) for v in batch["views"]]
            mu, _ = self._encode(views_dev)
            all_mu.append(mu.cpu())
        mu_all = torch.cat(all_mu, dim=0)
        return [mu_all.numpy()]
