"""Deep CCA with the Eckart-Young loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cca_zoo._utils._ey import prefix_weights
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

    The loss is invariant to rotating the components, and with flexible
    encoders it ranks the cross-view-predictable functions of the data, which
    need not be the factors: functions of a very stable factor, such as its
    square, can outrank a less stable factor and fill components that factor
    needed. ``ordered=True`` breaks the rotation invariance, as for
    :class:`~cca_zoo.linear.gradient.CCAEY`, but not this crowding; see
    :func:`~cca_zoo.model_selection.nonlinear_redundancy` to check for it, and
    :class:`~cca_zoo.deep.LeJEPA` with ``ordered=True`` for a loss that also
    controls the embedding's distribution.

    Args:
        n_components: Number of latent dimensions.
        encoders: One module per view.
        learning_rate: Adam learning rate. Default is 1e-3.
        ordered: Whether to average the loss over every prefix of the
            components, so that component j is the j-th canonical direction of
            the encoders' outputs. Default is False.

    References:
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DCCAEY
        >>> model = DCCAEY(n_components=4, encoders=[nn.Linear(10, 4), nn.Linear(8, 4)])
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        learning_rate: float = 1e-3,
        ordered: bool = False,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.ordered = ordered

    def _prefix_weights(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The loss's weights on the diagonal of ``C`` and on ``V``'s entries.

        All ones for the plain loss, and :func:`prefix_weights` when ``ordered``,
        on the dtype and device of ``like``.
        """
        k = self.n_components
        d, entry = prefix_weights(k) if self.ordered else (np.ones(k), np.ones((k, k)))
        return (
            torch.as_tensor(d, dtype=like.dtype, device=like.device),
            torch.as_tensor(entry, dtype=like.dtype, device=like.device),
        )

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
        d, entry = self._prefix_weights(c)
        rewards = 2.0 * (d * c.diagonal()).sum()
        penalties = (entry * v * v_ind.T).sum()
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
