"""Deep variational CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch


def _gaussian_nll(
    views: list[torch.Tensor], reconstructions: list[torch.Tensor]
) -> torch.Tensor:
    """Gaussian negative log-likelihood per sample, up to a constant.

    Each view has unit variance.

    Squared errors are summed over features, as the ELBO requires; a mean over
    features would weight the KL terms by the number of features.
    """
    return (
        0.5
        * torch.stack(
            [(r - x).pow(2).sum(dim=1).mean() for x, r in zip(views, reconstructions)]
        ).sum()
    )


def _sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """A reparameterised sample from a diagonal Gaussian."""
    return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)


def _kl(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL divergence from a diagonal Gaussian to the standard normal, per sample."""
    return -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]


def _split(
    out: torch.Tensor, k: int, encoder: str, width: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and log-variance from an encoder output of width ``2 * k``."""
    if out.shape[1] != 2 * k:
        raise ValueError(
            f"{encoder} returned {out.shape[1]} outputs; it needs "
            f"2 * {width} = {2 * k}."
        )
    return out[:, :k], out[:, k:]


class DVCCA(BaseDeep):
    r"""Deep variational CCA: a latent inferred from one view generates all of them.

    With prior $z \sim \mathcal{N}(0, I)$, decoders $p(x_i \mid z)$ for every
    view and an encoder $q(z \mid x_1) = \mathcal{N}(\mu, \sigma^2)$ of the
    first view, training minimises the negative ELBO

    $$
    \mathcal{L} = \sum_i \tfrac12 \lVert x_i - \text{decoder}_i(z) \rVert^2
        + \mathrm{KL}(q(z \mid x_1) \,\|\, \mathcal{N}(0, I)).
    $$

    Unlike the other deep models there is one encoding, so ``trainer.predict``
    returns a single array, the posterior mean $\mu$.

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
        return _split(
            self.encoders[0](views[0]), self.n_components, "The encoder", "n_components"
        )

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
        z = _sample(mu, log_var)
        reconstruction = _gaussian_nll(views, [dec(z) for dec in self.decoders])
        kl = _kl(mu, log_var)
        return {
            "objective": reconstruction + kl,
            "reconstruction": reconstruction,
            "kl": kl,
        }

    def predict_step(self, batch: Batch, batch_idx: int) -> list[torch.Tensor]:
        """The posterior mean of a batch, as a one-element list."""
        return self.forward(batch["views"])


class DVCCAPrivate(DVCCA):
    r"""DVCCA with a private latent variable per view as well as the shared one.

    The shared $z$ is inferred from the first view, $q(z \mid x_1)$, and each
    view's private $h_i$ from that view, $q(h_i \mid x_i)$; view $i$ is decoded
    from $[z, h_i]$. All latents have standard normal priors, and training
    minimises the negative ELBO

    $$
    \mathcal{L} = \sum_i \tfrac12 \lVert x_i - \text{decoder}_i(z, h_i) \rVert^2
        + \mathrm{KL}(q(z \mid x_1) \,\|\, p(z))
        + \sum_i \mathrm{KL}(q(h_i \mid x_i) \,\|\, p(h_i)).
    $$

    The private variables take up view-specific variation, leaving $z$ the
    shared part. ``trainer.predict`` returns the posterior mean of $z$;
    :meth:`private_means` gives each view's private posterior mean.

    Args:
        n_components: Dimension of the shared latent.
        encoder: Module mapping the first view to ``2 * n_components`` outputs,
            the mean then the log-variance of $z$.
        private_encoders: One module per view mapping it to ``2 * n_private``
            outputs, the mean then the log-variance of its $h_i$.
        decoders: One module per view mapping ``n_components + n_private``
            inputs, $[z, h_i]$, back to that view.
        n_private: Dimension of each private latent.
        learning_rate: Adam learning rate. Default is 1e-3.

    References:
        Wang, W., Yan, X., Lee, H., & Livescu, K. (2016). Deep variational
        canonical correlation analysis. arXiv:1610.03454.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DVCCAPrivate
        >>> model = DVCCAPrivate(
        ...     n_components=4,
        ...     encoder=nn.Linear(10, 8),
        ...     private_encoders=[nn.Linear(10, 4), nn.Linear(6, 4)],
        ...     decoders=[nn.Linear(6, 10), nn.Linear(6, 6)],
        ...     n_private=2,
        ... )
    """

    def __init__(
        self,
        n_components: int,
        encoder: nn.Module,
        private_encoders: list[nn.Module],
        decoders: list[nn.Module],
        n_private: int,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoder=encoder,
            decoders=decoders,
            learning_rate=learning_rate,
        )
        self.private_encoders = nn.ModuleList(private_encoders)
        self.n_private = n_private

    def _private_posteriors(
        self, views: list[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Mean and log-variance of each view's ``q(h_i | x_i)``."""
        return [
            _split(enc(v), self.n_private, "A private encoder", "n_private")
            for enc, v in zip(self.private_encoders, views)
        ]

    def private_means(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Each view's private posterior mean, shape (batch_size, n_private)."""
        return [mu for mu, _ in self._private_posteriors(views)]

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The negative ELBO of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "reconstruction", "kl", "private_kl"}``.
        """
        views = batch["views"]
        mu, log_var = self._posterior(views)
        z = _sample(mu, log_var)
        private = self._private_posteriors(views)
        reconstruction = _gaussian_nll(
            views,
            [
                dec(torch.cat([z, _sample(m, lv)], dim=1))
                for dec, (m, lv) in zip(self.decoders, private)
            ],
        )
        kl = _kl(mu, log_var)
        private_kl = torch.stack([_kl(m, lv) for m, lv in private]).sum()
        return {
            "objective": reconstruction + kl + private_kl,
            "reconstruction": reconstruction,
            "kl": kl,
            "private_kl": private_kl,
        }
