"""DCCAE — Deep CCA with Autoencoders (Wang 2015)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import BaseDeep
from cca_zoo.deep.objectives import CCALoss


class DCCAE(BaseDeep):
    r"""Deep CCA with Autoencoders.

    Extends DCCA by adding per-view reconstruction losses.  The total
    objective is a convex combination of the CCA loss and the summed
    MSE reconstruction losses:

    $$
    \mathcal{L} = (1 - \lambda) \, \mathcal{L}_{\text{CCA}}(z_1, \dots, z_V)
        + \lambda \sum_i \operatorname{MSE}\!\bigl(x_i,\ \text{decoder}_i(z_i)\bigr)
    $$

    where $\mathcal{L}_{\text{CCA}}$ defaults to
    :class:`~cca_zoo.deep.objectives.CCALoss`. When $\lambda = 0$ the
    model reduces to :class:`~cca_zoo.deep.DCCA`; when $\lambda = 1$
    it is a pure autoencoder.

    References:
        Wang, W., et al. "On deep multi-view representation learning."
        ICML 2015.

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects mapping each
            view to the latent space.
        decoders: List of :class:`torch.nn.Module` objects mapping the
            latent space back to each view's input space.
        lam: Weight for the reconstruction term.  Must be in [0, 1].
            When 0 the model reduces to DCCA; when 1 it is a pure
            autoencoder. Default is 0.5.
        objective: Differentiable CCA loss operating on a list of latent
            tensors.  Defaults to :class:`~cca_zoo.deep.objectives.CCALoss`.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Ridge regularisation for the CCA loss. Default is 1e-6.

    Raises:
        ValueError: If ``lam`` is not in [0, 1].

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1, enc2 = nn.Linear(10, 4), nn.Linear(8, 4)
        >>> dec1, dec2 = nn.Linear(4, 10), nn.Linear(4, 8)
        >>> model = DCCAE(
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
        lam: float = 0.5,
        objective: nn.Module | None = None,
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        if lam < 0.0 or lam > 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}.")
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.lam = lam
        self.decoders = nn.ModuleList(decoders)
        self.objective: nn.Module = CCALoss(eps=eps) if objective is None else objective

    def _decode(self, representations: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode latent representations back to input space.

        Args:
            representations: List of latent tensors, each of shape
                (batch_size, latent_dimensions).

        Returns:
            List of reconstructed tensors, each matching the
            corresponding view's input shape.
        """
        return [dec(z) for dec, z in zip(self.decoders, representations)]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the DCCAE objective (CCA + reconstruction).

        This method does not have access to the original views for
        reconstruction.  Override ``training_step`` or call
        :meth:`_full_loss` if reconstruction targets are needed.

        Args:
            representations: Encoded views from the current batch.
            independent_representations: Unused.

        Returns:
            Dictionary with key ``"objective"`` containing the CCA loss
            (reconstruction is not computed here without raw views).
        """
        return {"objective": self.objective(representations)}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step that includes reconstruction loss.

        Args:
            batch: Dictionary with key ``"views"`` (list of tensors).
            batch_idx: Batch index (unused).

        Returns:
            Scalar loss tensor.
        """
        views = batch["views"]
        representations = self(views)
        reconstructions = self._decode(representations)

        cca_loss = self.objective(representations)
        recon_loss = torch.stack(
            [F.mse_loss(x, r) for x, r in zip(views, reconstructions)]
        ).sum()
        objective = (1.0 - self.lam) * cca_loss + self.lam * recon_loss

        loss_dict = {
            "objective": objective,
            "cca": cca_loss,
            "reconstruction": recon_loss,
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
