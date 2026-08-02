"""DCCA_EY — Deep CCA with EigenGame / Eckart-Young objective."""

from __future__ import annotations

import torch

from cca_zoo.deep._dcca import DCCA


def _cca_cv(
    representations: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the averaged cross-covariance and auto-covariance matrices.

    Args:
        representations: List of tensors each of shape
            (batch_size, latent_dimensions).

    Returns:
        Tuple ``(C, V)`` where C is the mean pairwise cross-covariance
        and V is the mean auto-covariance, both of shape
        (latent_dimensions, latent_dimensions).
    """
    k = representations[0].shape[1]
    device = representations[0].device
    c = torch.zeros(k, k, device=device)
    v = torch.zeros(k, k, device=device)
    n_views = len(representations)
    for zi in representations:
        zi_c = zi - zi.mean(dim=0)
        v = v + (zi_c.T @ zi_c) / (zi_c.shape[0] - 1)
        for zj in representations:
            zj_c = zj - zj.mean(dim=0)
            cross = (zi_c.T @ zj_c) / (zi_c.shape[0] - 1)
            c = c + cross
    c = c / n_views
    v = v / n_views
    return c, v


class DCCA_EY(DCCA):
    r"""DCCA using the EigenGame / Eckart-Young (EY) objective.

    For $M$ views with embeddings $Z_1, \dots, Z_M$, define the
    mean pairwise cross-covariance $C$ and mean auto-covariance
    $V$ (see :mod:`cca_zoo._utils._ey` for the full definitions).
    The EY loss minimised is:

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(VV)
    $$

    This is an *unconstrained* stand-in for CCA — unlike the exact
    eigendecomposition solution, it requires no manifold projection and is
    a stationary point exactly at the canonical directions, making it
    suitable for mini-batch gradient descent. When
    ``independent_representations`` are provided, the $\operatorname{tr}(VV)$
    penalty becomes $\operatorname{tr}(V V_{\text{ind}})$ to decouple
    estimation of the two quantities (as in the EigenGame formulation).

    References:
        Chapman, J., Aguila, A. L., & Wells, L. "A Generalised EigenGame
        with Extensions to Multiview Representation Learning."
        arXiv:2211.11323 (2022).

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        objective: Ignored; the EY objective is fixed for this class.
            Accepted for API compatibility but overridden internally.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Regularisation for numerical stability. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = DCCA_EY(latent_dimensions=4, encoders=[enc1, enc2])
    """

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the EigenGame / Eckart-Young CCA loss.

        Args:
            representations: Encoded views from the current batch.
            independent_representations: Optional second set of encodings
                for the EigenGame penalty term.  When provided the penalty
                is ``tr(V @ V_ind)`` instead of ``tr(V @ V)``.

        Returns:
            Dictionary with keys ``"objective"``, ``"rewards"``, and
            ``"penalties"``.
        """
        c, v = _cca_cv(representations)
        rewards = torch.trace(2.0 * c)
        if independent_representations is None:
            penalties = torch.trace(v @ v)
        else:
            _, v_ind = _cca_cv(independent_representations)
            penalties = torch.trace(v @ v_ind)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
