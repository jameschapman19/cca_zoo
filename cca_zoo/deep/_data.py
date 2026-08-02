"""MultiviewDataset — minimal Dataset wrapper for array-backed views."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset


class MultiviewDataset(Dataset[dict[str, list[torch.Tensor]]]):
    r"""Wraps per-view arrays into the batch format :class:`BaseDeep` expects.

    :meth:`~cca_zoo.deep.BaseDeep.training_step` and ``validation_step``
    read each batch as a dict with a ``"views"`` key holding a list of
    per-view tensors — plain :class:`torch.utils.data.TensorDataset`
    yields tuples instead, which does not match that contract.  This class
    is a small, optional convenience for the common case of already having
    each view as a single in-memory array.

    For anything more custom (lazy loading, augmentation, lists of file
    paths, lists of paired NOI-style samples, ...), write your own
    :class:`~torch.utils.data.Dataset` whose ``__getitem__`` returns a dict
    of that same shape — that is the actual contract, not this class.

    Args:
        views: List of array-likes, each of shape (n_samples, n_features_i).
            All must have the same number of samples.

    Example:
        >>> import numpy as np
        >>> from torch.utils.data import DataLoader
        >>> from cca_zoo.deep import MultiviewDataset
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 10)).astype("float32")
        >>> X2 = rng.standard_normal((100, 8)).astype("float32")
        >>> loader = DataLoader(MultiviewDataset([X1, X2]), batch_size=32)
        >>> batch = next(iter(loader))
        >>> list(batch.keys())
        ['views']
    """

    def __init__(self, views: list[ArrayLike]) -> None:
        self.views: list[torch.Tensor] = [
            torch.as_tensor(np.asarray(v), dtype=torch.float32) for v in views
        ]

    def __len__(self) -> int:
        """Number of samples (rows) shared by every view."""
        return int(self.views[0].shape[0])

    def __getitem__(self, index: int) -> dict[str, list[torch.Tensor]]:
        """Return the ``index``-th sample as a ``{"views": [...]}`` dict.

        Args:
            index: Sample index.

        Returns:
            Dictionary with key ``"views"``: a list of per-view tensors,
            each of shape (n_features_i,).
        """
        return {"views": [v[index] for v in self.views]}
