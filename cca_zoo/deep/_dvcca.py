"""DVCCA — Deep Variational CCA (Wang 2016)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import BaseDeep


class DVCCA(BaseDeep):
    r"""Deep Variational Canonical Correlation Analysis.

    A variational autoencoder framework for multiview data.  Each
    encoder maps a view to a 2 * latent_dimensions output, which is
    split into a posterior mean mu and log-variance log_var.  A shared
    latent code z is sampled via the reparameterisation trick and then
    decoded to reconstruct all views.

    The training objective is the negative ELBO:

    $$
    \mathcal{L} = \sum_i \operatorname{MSE}\!\bigl(x_i,\ \text{decoder}_i(z)\bigr)
        + \mathrm{KL}\!\bigl(q(z \mid X) \,\|\, p(z)\bigr)
    $$

    where the approximate posterior aggregates all views' encoders and the
    prior is a standard normal:

    $$
    q(z \mid X) = \mathcal{N}\!\Bigl(
        \textstyle\sum_i \mu_i,\
        \operatorname{diag}\bigl(\exp(\textstyle\sum_i \log\sigma_i^2)\bigr)
    \Bigr), \qquad p(z) = \mathcal{N}(0, I)
    $$

    References:
        Wang, W., et al. "Deep variational canonical correlation
        analysis." arXiv:1610.03454 (2016).

    Args:
        latent_dimensions: Dimensionality of the latent space.
        encoders: List of :class:`torch.nn.Module` objects each mapping
            a view to a vector of size 2 * latent_dimensions (first half
            is mu, second half is log_var).
        decoders: List of :class:`torch.nn.Module` objects mapping the
            latent vector back to each view's input space.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Regularisation for numerical stability. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> # Encoders output 2 * latent_dimensions
        >>> enc1 = nn.Linear(10, 8)
        >>> enc2 = nn.Linear(10, 8)
        >>> dec1 = nn.Linear(4, 10)
        >>> dec2 = nn.Linear(4, 10)
        >>> model = DVCCA(
        ...     latent_dimensions=4,
        ...     encoders=[enc1, enc2],
        ...     decoders=[dec1, dec2],
        ... )
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        decoders: list[nn.Module],
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.decoders = nn.ModuleList(decoders)

    def _encode(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all views and aggregate their posterior parameters.

        Each encoder outputs 2 * latent_dimensions values; the first
        half is mu and the second half is log_var.  The shared posterior
        is formed by summing across views.

        Args:
            views: List of input tensors, one per view.

        Returns:
            Tuple ``(mu, log_var)`` each of shape
            (batch_size, latent_dimensions).
        """
        k = self.latent_dimensions
        mu_sum = torch.zeros(views[0].shape[0], k, device=views[0].device)
        lv_sum = torch.zeros(views[0].shape[0], k, device=views[0].device)
        for enc, v in zip(self.encoders, views):
            out = enc(v)
            mu_sum = mu_sum + out[:, :k]
            lv_sum = lv_sum + out[:, k:]
        return mu_sum, lv_sum

    def _reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z via the reparameterisation trick.

        Args:
            mu: Posterior mean, shape (batch_size, latent_dimensions).
            log_var: Posterior log-variance, shape
                (batch_size, latent_dimensions).

        Returns:
            Sampled latent code of the same shape as ``mu``.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Decode the latent code to reconstruct all views.

        Args:
            z: Latent tensor of shape (batch_size, latent_dimensions).

        Returns:
            List of reconstructed tensors, one per view.
        """
        return [dec(z) for dec in self.decoders]

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode views and return a list of latent representations (mu).

        At inference time this returns the posterior mean as the point
        estimate of the latent code for each view independently.

        Args:
            views: List of input tensors, one per view.

        Returns:
            List with one tensor of shape (batch_size, latent_dimensions)
            representing the shared posterior mean.
        """
        mu, _ = self._encode(views)
        return [mu]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the ELBO loss (used during validation via BaseDeep).

        Note: Reconstruction requires access to original views; for
        training the full ELBO is computed in :meth:`training_step`.

        Args:
            representations: Unused here; the method returns zero so
                that the validation step in BaseDeep does not crash.
            independent_representations: Unused.

        Returns:
            Dictionary with ``"objective"`` set to 0.
        """
        return {"objective": torch.tensor(0.0)}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step computing the full negative ELBO.

        Args:
            batch: Dictionary with key ``"views"`` (list of tensors).
            batch_idx: Batch index (unused).

        Returns:
            Scalar loss tensor (negative ELBO).
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
        """Project all samples to the shared latent space via posterior mean.

        Args:
            loader: DataLoader yielding batches with a ``"views"`` key.

        Returns:
            List with one numpy array of shape (n_samples, latent_dimensions).
        """
        self.eval()
        all_mu: list[torch.Tensor] = []
        for batch in loader:
            views_dev = [v.to(self.device) for v in batch["views"]]
            mu, _ = self._encode(views_dev)
            all_mu.append(mu.cpu())
        mu_all = torch.cat(all_mu, dim=0)
        return [mu_all.numpy()]
