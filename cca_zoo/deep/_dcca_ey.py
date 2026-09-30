"""Deep CCA with the Eckart-Young loss."""

from __future__ import annotations

import torch

from cca_zoo.deep._base import BaseDeep, Batch


def _cca_cv(
    representations: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean pairwise cross-covariance and mean auto-covariance, each (k, k)."""
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


class DCCAEY(BaseDeep):
    r"""Deep CCA by minimising the Eckart-Young loss.

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V),
    $$

    (:mod:`cca_zoo._utils._ey`), which is unconstrained and so suits
    mini-batches. With independent representations the penalty is
    $\operatorname{tr}(V V_{\text{ind}})$, an unbiased estimate.

    Args:
        n_components: Number of latent dimensions.
        encoders: One module per view.
        learning_rate: Adam learning rate. Default is 1e-3.

    References:
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DCCAEY
        >>> model = DCCAEY(n_components=4, encoders=[nn.Linear(10, 4), nn.Linear(8, 4)])
    """

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The EY loss of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors and, optionally,
                ``"independent_views"`` from an independent batch, which
                give an unbiased estimate of the penalty.

        Returns:
            ``{"objective", "rewards", "penalties"}``.
        """
        c, v = _cca_cv(self(batch["views"]))
        independent = batch.get("independent_views")
        v_ind = v if independent is None else _cca_cv(self(independent))[1]
        rewards = torch.trace(2.0 * c)
        penalties = torch.trace(v @ v_ind)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
