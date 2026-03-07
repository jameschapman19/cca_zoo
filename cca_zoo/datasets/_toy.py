"""Toy real-world dataset loaders for multiview CCA experiments."""

from __future__ import annotations

import numpy as np


def load_linnerud() -> tuple[np.ndarray, np.ndarray]:
    """Load the Linnerud dataset as two views.

    The Linnerud dataset (from scikit-learn) contains two sets of
    measurements on 20 middle-aged men: exercise performance and
    physiological measurements.  This function returns them as a pair of
    numpy arrays suitable for two-view CCA.

    Returns:
        Tuple ``(exercise, physiological)``. ``exercise`` is shape (20, 3)
        with chin-up, sit-up, and jump counts. ``physiological`` is shape
        (20, 3) with weight, waist, and pulse measurements.

    Example:
        >>> X1, X2 = load_linnerud()
        >>> X1.shape
        (20, 3)
        >>> X2.shape
        (20, 3)
    """
    from sklearn.datasets import load_linnerud as _load

    dataset = _load()
    # dataset.data = exercise, dataset.target = physiological
    return np.asarray(dataset.data), np.asarray(dataset.target)


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """Load the Wisconsin breast cancer dataset split into two feature views.

    The 30 features of the Wisconsin Diagnostic Breast Cancer dataset are
    split into two equal halves of 15 features each, providing a simple
    two-view dataset for benchmarking multiview methods.

    Returns:
        Tuple ``(view1, view2)`` where each array has shape (569, 15).

    Example:
        >>> X1, X2 = load_breast_cancer()
        >>> X1.shape
        (569, 15)
        >>> X2.shape
        (569, 15)
    """
    from sklearn.datasets import load_breast_cancer as _load

    dataset = _load()
    x: np.ndarray = np.asarray(dataset.data)
    midpoint = x.shape[1] // 2
    return x[:, :midpoint], x[:, midpoint:]
