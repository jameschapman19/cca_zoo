"""Deep partial CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep._dcca_ey import _cca_cv


def _partialled(z: torch.Tensor, f: torch.Tensor, eps: float) -> torch.Tensor:
    """``f`` less its least-squares regression on ``z``, both centred."""
    eye = eps * torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
    residual: torch.Tensor = f - z @ torch.linalg.solve(z.T @ z + eye, z.T @ f)
    return residual


class DPCCA(BaseDeep):
    r"""Deep partial CCA: correlate the views after conditioning on a shared variable.

    Each batch's encodings $F_i$ are partialled on the conditioning variable
    $Z$ (``batch["partials"]``) by the batch's own regression,
    $F_i|Z = F_i - Z (Z^\top Z)^{-1} Z^\top F_i$, and minimise the
    Eckart-Young loss of :class:`DCCAEY`,

    $$
    \mathcal{L} = -2 \operatorname{tr}(C_{|Z}) + \operatorname{tr}(V_{|Z} V_{|Z}),
    $$

    whose $C_{|Z}$ and $V_{|Z}$ are then the batch's mean pairwise partial
    cross-covariance and mean partial auto-covariance,
    $\Sigma_{FF} - \Sigma_{FZ} \Sigma_{ZZ}^{-1} \Sigma_{ZF}$. Gradients flow
    through the regression, so this is the derivative of the partial
    covariances' loss.

    ``partial_encoder=None`` uses $Z$ as given; a module encodes it, trained to
    explain the encodings by least squares, so that partialling removes all
    the encoded $Z$ can explain. $Z$ is needed only for training: prediction
    encodes the views alone.

    Rotman et al. train by nonlinear orthogonal iterations, regressing each
    view onto the other's whitened partialled encoding, with running
    estimates of the covariances across batches; the EY loss needs neither
    whitening nor running estimates, and suits mini-batch training (Chapman et
    al., 2024). Their encoded variant also trains the partial encoder on the
    correlation loss, which rewards it for failing to explain the confound;
    here it minimises the partialling residual instead. With two views the
    model is theirs; more views are handled as in :class:`DCCAEY`.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        partial_encoder: Module encoding the conditioning variable, or None to
            use it as given. Default is None.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Ridge added to $Z^\top Z$ before inversion. Default is 1e-6.

    Raises:
        ValueError: If a training batch has no ``"partials"``.

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
        learning_rate: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.partial_encoder = partial_encoder
        self.eps = eps

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
        c, v = _cca_cv([_partialled(z.detach(), f, self.eps) for f in encodings])
        rewards = torch.trace(2.0 * c)
        penalties = torch.trace(v @ v)
        terms = {"rewards": rewards, "penalties": penalties}
        objective = -rewards + penalties
        if self.partial_encoder is not None:
            # The partial encoder learns to explain the encodings, held fixed, so
            # partialling removes all it can; trained on the EY loss it would
            # instead learn to leave the confound in.
            residual = torch.stack(
                [_partialled(z, f.detach(), self.eps).pow(2).mean() for f in encodings]
            ).sum()
            terms["residual"] = residual
            objective = objective + residual
        return {"objective": objective, **terms}
