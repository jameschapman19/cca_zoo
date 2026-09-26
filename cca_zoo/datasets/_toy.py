"""Toy real-world multiview datasets."""

from __future__ import annotations

import numpy as np


def load_linnerud() -> tuple[np.ndarray, np.ndarray]:
    """The Linnerud data as exercise and physiological views.

    Returns:
        ``(exercise, physiological)``, each of shape (20, 3).

    Examples:
        >>> from cca_zoo.datasets import load_linnerud
        >>> X1, X2 = load_linnerud()
        >>> X1.shape, X2.shape
        ((20, 3), (20, 3))
    """
    from sklearn.datasets import load_linnerud as _load

    dataset = _load()
    # dataset.data = exercise, dataset.target = physiological
    return np.asarray(dataset.data), np.asarray(dataset.target)


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """The Wisconsin breast cancer features split into two 15-feature views.

    Returns:
        ``(view1, view2)``, each of shape (569, 15).

    Examples:
        >>> from cca_zoo.datasets import load_breast_cancer
        >>> X1, X2 = load_breast_cancer()
        >>> X1.shape, X2.shape
        ((569, 15), (569, 15))
    """
    from sklearn.datasets import load_breast_cancer as _load

    dataset = _load()
    x: np.ndarray = np.asarray(dataset.data)
    midpoint = x.shape[1] // 2
    return x[:, :midpoint], x[:, midpoint:]
