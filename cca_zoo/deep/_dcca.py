"""Deep CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch
from cca_zoo.deep.objectives import CCALoss


class _ObjectiveModel(BaseDeep):
    """A deep model minimising a loss module of the encodings, ``objective``."""

    objective: nn.Module

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """``{"objective": objective(encodings)}``."""
        return {"objective": self.objective(self(batch["views"]))}


class DCCA(_ObjectiveModel):
    r"""Deep CCA of two views: encoders trained to maximise canonical correlation.

    Minimises, per mini-batch,

    $$
    \mathcal{L} = -\bigl\| \Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2} \bigr\|_F^2
    $$

    (:class:`~cca_zoo.deep.objectives.CCALoss`). As linear CCA generalises to
    MCCA, GCCA and TCCA, it generalises to more views as :class:`DMCCA`,
    :class:`DGCCA` and :class:`DTCCA`.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Ridge of the within-view covariances. Default is 1e-6.

    Raises:
        ValueError: If there are not two encoders.

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
        >>> model.fit_cca(loader)
        >>> Z1, Z2 = (torch.cat(z) for z in zip(*trainer.predict(model, loader)))
        >>> Z1.shape
        torch.Size([64, 2])
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        learning_rate: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        if len(encoders) != 2:
            raise ValueError(
                f"DCCA is defined for two views, got {len(encoders)}; use DMCCA, "
                "DGCCA or DTCCA for more."
            )
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.eps = eps
        self.objective = CCALoss(eps=eps)
