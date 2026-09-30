"""Deep partial CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch


class DPCCA(BaseDeep):
    r"""Deep partial CCA: correlate the views after conditioning on a shared variable.

    The Eckart-Young loss of :class:`DCCAEY` on the views' partial covariances
    given a conditioning variable $Z$ (``batch["partials"]``). For the
    stacked encodings $F$ of a batch,

    $$
    \Sigma_{FF|Z} = \Sigma_{FF} - \Sigma_{FZ} \Sigma_{ZZ}^{-1} \Sigma_{ZF},
    \qquad
    \mathcal{L} = -2 \operatorname{tr}(C_{|Z}) + \operatorname{tr}(V_{|Z} V_{|Z}),
    $$

    with $C_{|Z}$ the mean over views of the sum of each view's blocks of
    $\Sigma_{FF|Z}$, and $V_{|Z}$ the mean of its diagonal blocks, as
    :class:`DCCAEY` forms them from the covariances.

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
        n_components: Number of latent dimensions.
        encoders: One module per view.
        partial_encoder: Module encoding the conditioning variable, or None to
            use it as given. Default is None.
        learning_rate: Adam learning rate. Default is 1e-3.
        reg_covar: Non-negative regularisation added to the diagonal of each
            covariance, as scikit-learn's ``GaussianMixture``. Default is 1e-6.

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
        reg_covar: float = 1e-6,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.partial_encoder = partial_encoder
        self.reg_covar = reg_covar

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
        """The EY loss of the encodings' partial covariances and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors and the
                conditioning variable under ``"partials"``.

        Returns:
            ``{"objective", "rewards", "penalties"}``, and ``"residual"``, the
            partial encoder's loss, when there is one.
        """
        z = self._conditioning(batch)
        f = torch.cat([f - f.mean(dim=0) for f in self(batch["views"])], dim=1)
        n, m, k = f.shape[0], len(batch["views"]), self.n_components
        eye = self.reg_covar * torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
        # The partial covariance of the encodings given Z, held fixed.
        z_fixed = z.detach()
        zf = z_fixed.T @ f
        partial = (
            f.T @ f - zf.T @ torch.linalg.solve(z_fixed.T @ z_fixed + eye, zf)
        ) / (n - 1)
        blocks = partial.reshape(m, k, m, k)
        c = blocks.sum(dim=(0, 2)) / m
        v = torch.diagonal(blocks, dim1=0, dim2=2).sum(dim=-1) / m
        rewards = torch.trace(2.0 * c)
        penalties = torch.trace(v @ v)
        terms = {"rewards": rewards, "penalties": penalties}
        objective = -rewards + penalties
        if self.partial_encoder is not None:
            # The partial encoder learns to explain the encodings, held fixed, so
            # partialling removes all it can; trained on the EY loss it would
            # instead learn to leave the confound in. Its loss is the mean
            # squared residual, the trace of the partial second moment.
            f_fixed = f.detach()
            zf = z.T @ f_fixed
            residual = torch.trace(
                f_fixed.T @ f_fixed - zf.T @ torch.linalg.solve(z.T @ z + eye, zf)
            ) / (n * k)
            terms["residual"] = residual
            objective = objective + residual
        return {"objective": objective, **terms}
