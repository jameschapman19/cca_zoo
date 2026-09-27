"""Dataset wrapper for in-memory views."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset


class MultiviewDataset(Dataset[dict[str, Any]]):
    """Dataset of in-memory views, yielding ``{"views": [...]}`` samples.

    The deep models read batches as dictionaries with a ``"views"`` list, and
    :class:`DPCCA` also a ``"partials"`` tensor; any dataset returning that
    shape works.

    Args:
        views: Arrays of shape (n_samples, n_features_i), one per view.
        partials: Conditioning variable for :class:`DPCCA`, shape
            (n_samples, n_partials). Default is None.

    Examples:
        >>> import numpy as np
        >>> from torch.utils.data import DataLoader
        >>> from cca_zoo.deep import MultiviewDataset
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 10)).astype("float32")
        >>> X2 = rng.standard_normal((100, 8)).astype("float32")
        >>> batch = next(iter(DataLoader(MultiviewDataset([X1, X2]), batch_size=32)))
        >>> [v.shape for v in batch["views"]]
        [torch.Size([32, 10]), torch.Size([32, 8])]
    """

    def __init__(
        self, views: list[ArrayLike], partials: ArrayLike | None = None
    ) -> None:
        self.views: list[torch.Tensor] = [
            torch.as_tensor(np.asarray(v), dtype=torch.float32) for v in views
        ]
        self.partials = (
            None
            if partials is None
            else torch.as_tensor(np.asarray(partials), dtype=torch.float32)
        )

    def __len__(self) -> int:
        """Number of samples."""
        return int(self.views[0].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Sample ``index`` as ``{"views": [tensor per view]}``, with its partials."""
        item: dict[str, Any] = {"views": [v[index] for v in self.views]}
        if self.partials is not None:
            item["partials"] = self.partials[index]
        return item
