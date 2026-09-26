"""Deep CCA with the Eckart-Young loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._dcca import DCCA


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


class DCCAEY(DCCA):
    r"""Deep CCA by minimising the Eckart-Young loss.

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V),
    $$

    (:mod:`cca_zoo._utils._ey`), which is unconstrained and so suits
    mini-batches. With independent representations the penalty is
    $\operatorname{tr}(V V_{\text{ind}})$, an unbiased estimate.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        learning_rate: Adam learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.

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
        max_epochs: int = 100,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
        )

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """The EY loss of a batch and its terms.

        Args:
            representations: One encoded tensor per view.
            independent_representations: Encodings of an independent batch for
                the penalty. Default is None.

        Returns:
            ``{"objective", "rewards", "penalties"}``.
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
