"""Deep partial CCA."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep.objectives import _inv_sqrtm


def _covariance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cross-covariance of two centred batches."""
    return a.T @ b / a.shape[0]


class DPCCA(BaseDeep):
    r"""Deep partial CCA: correlate the views after conditioning on a shared variable.

    With encodings $F_i$ and a conditioning variable $Z$ (``batch["partials"]``),
    each encoding is partialled on $Z$ using running covariance estimates,
    $F_i|Z = F_i - Z \Sigma_{ZZ}^{-1} \Sigma_{Z F_i}$, with conditional covariance
    $\Sigma_{ii|Z} = \Sigma_{ii} - \Sigma_{Z F_i}^\top \Sigma_{ZZ}^{-1} \Sigma_{Z F_i}$.
    Training is by nonlinear orthogonal iterations: each view regresses onto the
    other's whitened conditional encoding, held fixed,

    $$
    \mathcal{L} = \sum_{i \neq j} \operatorname{MSE}\bigl(F_i|Z,\
        \operatorname{sg}(F_j|Z\, \Sigma_{jj|Z}^{-1/2})\bigr).
    $$

    ``partial_encoder=None`` uses $Z$ as given (the paper's variant A); a module
    encodes it, trained jointly (variant B). $Z$ is needed only for training:
    predictions use the views alone. With two views this is the published
    method; more views are summed over pairs.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        partial_encoder: Module encoding the conditioning variable, or None to
            use it as given. Default is None.
        rho: Weight of the previous running covariances in ``[0, 1)``. Default
            is 0.75.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Floor on the eigenvalues of inverted covariances. Default is 1e-6.

    Raises:
        ValueError: If ``rho`` is outside ``[0, 1)``, or a training batch has no
            ``"partials"``.

    References:
        Rotman, G., Vulić, I., & Reichart, R. (2018). Bridging languages
        through images with deep partial canonical correlation analysis. ACL.

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

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The DPCCA loss of a batch.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors and the
                conditioning variable under ``"partials"``.

        Returns:
            ``{"objective": loss}``.
        """
        if "partials" not in batch:
            raise ValueError(
                'DPCCA trains on batches with the conditioning variable "partials".'
            )
        partials = batch["partials"]
        z = partials if self.partial_encoder is None else self.partial_encoder(partials)
        z = z - z.mean(dim=0)
        encodings = [f - f.mean(dim=0) for f in self(batch["views"])]
        zz = self._running_covariance("zz", _covariance(z, z))
        eye = torch.eye(zz.shape[0], device=zz.device)
        zz_inv = torch.linalg.inv(zz.detach() + self.eps * eye)
        conditional, whitened = [], []
        for i, f in enumerate(encodings):
            zf = self._running_covariance(f"z{i}", _covariance(z, f))
            ff = self._running_covariance(f"{i}{i}", _covariance(f, f))
            f_given_z = f - z @ zz_inv @ zf
            ff_given_z = ff - zf.T @ zz_inv @ zf
            conditional.append(f_given_z)
            whitened.append(
                (f_given_z @ _inv_sqrtm(ff_given_z.detach(), self.eps)).detach()
            )
        mse = nn.functional.mse_loss
        objective = torch.stack(
            [
                mse(conditional[i], whitened[j])
                for i, j in itertools.permutations(range(len(encodings)), 2)
            ]
        ).sum()
        return {"objective": objective}
