"""Deep partial CCA."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep._dcca_ey import _cca_cv
from cca_zoo.linear._mcca import MCCA


def _covariance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cross-covariance of two centred batches."""
    return a.T @ b / (a.shape[0] - 1)


class DPCCA(BaseDeep):
    r"""Deep partial CCA: correlate the views after conditioning on a shared variable.

    Each encoding $F_i$ is partialled on a conditioning variable $Z$
    (``batch["partials"]``), $F_i|Z = F_i - Z \beta_i$ with
    $\beta_i = \Sigma_{ZZ}^{-1} \Sigma_{Z F_i}$, and the partialled encodings
    minimise the Eckart-Young loss of :class:`DCCAEY`,

    $$
    \mathcal{L} = -2 \operatorname{tr}(C_{|Z}) + \operatorname{tr}(V_{|Z} V_{|Z}),
    $$

    with $C_{|Z}$ and $V_{|Z}$ the mean pairwise cross-covariance and mean
    auto-covariance of the partialled encodings. $\Sigma_{ZZ}$ and
    $\Sigma_{Z F_i}$ are running estimates, as in the original method.
    ``partial_encoder=None`` uses $Z$ as given; a module encodes it, trained to
    explain the encodings by least squares, so that partialling removes all
    the encoded $Z$ can explain. $Z$ is needed only for training: the linear
    CCA applied at prediction is fitted on partialled training encodings and
    then applied to encodings of the views alone.

    Rotman et al. train by nonlinear orthogonal iterations, regressing each
    view onto the other's whitened partialled encoding; this implementation
    minimises the EY loss instead, which needs no whitening or inverse square
    roots and suits mini-batch training (Chapman et al., 2024). Their encoded
    variant also trains the partial encoder on the correlation loss, which
    rewards it for failing to explain the confound; here it minimises the
    partialling residual instead. With two views the model is theirs; more
    views are handled as in :class:`DCCAEY`.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        partial_encoder: Module encoding the conditioning variable, or None to
            use it as given. Default is None.
        rho: Weight of the previous running covariances in ``[0, 1)``. Default
            is 0.75.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Ridge added to $\Sigma_{ZZ}$ before inversion. Default is 1e-6.

    Raises:
        ValueError: If ``rho`` is outside ``[0, 1)``, or a training batch has no
            ``"partials"``.

    References:
        Rotman, G., Vulić, I., & Reichart, R. (2018). Bridging languages
        through images with deep partial canonical correlation analysis. ACL.

        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DPCCA
        >>> encoders = [nn.Linear(10, 4), nn.Linear(8, 4)]
        >>> model = DPCCA(n_components=4, encoders=encoders)
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        partial_encoder: nn.Module | None = None,
        rho: float = 0.75,
        learning_rate: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.partial_encoder = partial_encoder
        self.rho = rho
        self.eps = eps
        self._running: dict[str, torch.Tensor] = {}

    def _conditioning(self, batch: Batch) -> torch.Tensor:
        """The batch's centred conditioning variable, encoded if configured."""
        if "partials" not in batch:
            raise ValueError(
                'DPCCA trains on batches with the conditioning variable "partials".'
            )
        partials: torch.Tensor = batch["partials"]
        z = partials if self.partial_encoder is None else self.partial_encoder(partials)
        centred: torch.Tensor = z - z.mean(dim=0)
        return centred

    def _running_covariance(self, name: str, batch_cov: torch.Tensor) -> torch.Tensor:
        """Exponential moving average of a covariance, updated in training mode.

        The previous estimate is held fixed; gradients flow through the batch's
        share.
        """
        previous = self._running.get(name)
        if not self.training:
            return batch_cov if previous is None else previous
        current = (
            batch_cov
            if previous is None
            else self.rho * previous + (1.0 - self.rho) * batch_cov
        )
        self._running[name] = current.detach()
        return current

    def _partialled(
        self, z: torch.Tensor, encodings: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Each centred encoding with its regression on ``z`` removed."""
        zz = self._running_covariance("zz", _covariance(z, z)).detach()
        zz_inv = torch.linalg.inv(
            zz + self.eps * torch.eye(zz.shape[0], device=z.device)
        )
        return [
            f - z @ zz_inv @ self._running_covariance(f"z{i}", _covariance(z, f))
            for i, f in enumerate(encodings)
        ]

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The EY loss of the partialled encodings and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors and the
                conditioning variable under ``"partials"``.

        Returns:
            ``{"objective", "rewards", "penalties"}``, and ``"residual"``, the
            partial encoder's loss, when there is one.
        """
        z = self._conditioning(batch)
        encodings = [f - f.mean(dim=0) for f in self(batch["views"])]
        c, v = _cca_cv(self._partialled(z.detach(), encodings))
        rewards = torch.trace(2.0 * c)
        penalties = torch.trace(v @ v)
        terms = {"rewards": rewards, "penalties": penalties}
        objective = -rewards + penalties
        if self.partial_encoder is not None:
            # The partial encoder learns to explain the encodings, held fixed, so
            # partialling removes all it can; trained on the EY loss it would
            # instead learn to leave the confound in.
            eye = self.eps * torch.eye(z.shape[1], device=z.device)
            residual = torch.stack(
                [
                    (f - z @ torch.linalg.solve(z.T @ z + eye, z.T @ f)).pow(2).mean()
                    for f in (f.detach() for f in encodings)
                ]
            ).sum()
            terms["residual"] = residual
            objective = objective + residual
        return {"objective": objective, **terms}

    @torch.no_grad()
    def fit_cca(self, dataloader: Iterable[Batch]) -> None:
        """Fit the linear CCA that ``predict_step`` applies, on partialled encodings.

        The encodings of ``dataloader`` are partialled on its ``"partials"`` by
        least squares over the whole set, so the projection targets the
        correlation not explained by the conditioning variable.

        Args:
            dataloader: Batches with ``"views"`` and ``"partials"`` keys.
        """
        was_training = self.training
        self.eval()
        encodings: list[list[torch.Tensor]] = []
        conditioning: list[torch.Tensor] = []
        for batch in dataloader:
            batch = {k: _to(v, self.device) for k, v in batch.items()}
            encodings.append(self(batch["views"]))
            conditioning.append(self._conditioning(batch))
        self.train(was_training)
        z = torch.cat(conditioning).cpu().numpy()
        z = z - z.mean(axis=0)
        views = [torch.cat(view).cpu().numpy() for view in zip(*encodings)]
        means = [v.mean(axis=0) for v in views]
        partialled: list[ArrayLike] = [
            (v - m) - z @ np.linalg.lstsq(z, v - m, rcond=None)[0]
            for v, m in zip(views, means)
        ]
        cca = MCCA(n_components=self.n_components).fit(partialled)
        self.cca_weights.copy_(torch.as_tensor(np.stack(cca.weights_)))
        self.cca_means.copy_(torch.as_tensor(np.stack(means)))
        self.cca_fitted.fill_(True)


def _to(value: torch.Tensor | list[torch.Tensor], device: torch.device) -> object:
    """``value``, or each tensor in it, on ``device``."""
    if isinstance(value, list):
        return [v.to(device) for v in value]
    return value.to(device)
