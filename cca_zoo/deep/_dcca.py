"""Deep CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch, _require_two_views
from cca_zoo.deep.objectives import CCALoss


class DCCA(BaseDeep):
    r"""Deep CCA: neural encoders trained to maximise canonical correlation.

    By default minimises, per mini-batch,

    $$
    \mathcal{L} = -\bigl\| \Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2} \bigr\|_F^2
    $$

    (:class:`~cca_zoo.deep.objectives.CCALoss`), which is defined for two
    views; pass ``objective`` to use another loss, such as
    :class:`~cca_zoo.deep.objectives.MCCALoss` for more views.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        objective: Loss on the list of encodings; None uses ``CCALoss(eps)``.
            Default is None.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Ridge of the default loss. Default is 1e-6.

    Raises:
        ValueError: If ``objective`` is None and there are not two encoders.

    References:
        Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013). Deep
        canonical correlation analysis. ICML.

    Examples:
        >>> import lightning.pytorch as pl
        >>> import numpy as np
        >>> import torch
        >>> import torch.nn as nn
        >>> from torch.utils.data import DataLoader
        >>> from cca_zoo.deep import DCCA, MultiviewDataset
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((64, 10)).astype("float32")
        >>> X2 = rng.standard_normal((64, 8)).astype("float32")
        >>> loader = DataLoader(MultiviewDataset([X1, X2]), batch_size=32)
        >>> model = DCCA(n_components=2, encoders=[nn.Linear(10, 2), nn.Linear(8, 2)])
        >>> trainer = pl.Trainer(max_epochs=2, logger=False, enable_progress_bar=False,
        ...                      enable_checkpointing=False, enable_model_summary=False)
        >>> trainer.fit(model, loader)
        >>> Z1, Z2 = (torch.cat(z) for z in zip(*trainer.predict(model, loader)))
        >>> Z1.shape
        torch.Size([64, 2])
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        objective: nn.Module | None = None,
        learning_rate: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        if objective is None:
            _require_two_views(encoders, "CCALoss")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.eps = eps
        self.objective: nn.Module = CCALoss(eps=eps) if objective is None else objective

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """``{"objective": objective(encodings)}``."""
        return {"objective": self.objective(self(batch["views"]))}
