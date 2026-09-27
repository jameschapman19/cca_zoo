"""Toy real-world multiview datasets."""

from __future__ import annotations

import numpy as np
from sklearn import datasets
from sklearn.utils import Bunch


def load_linnerud(*, return_views: bool = False) -> Bunch | list[np.ndarray]:
    """The Linnerud data as exercise and physiological views.

    :func:`sklearn.datasets.load_linnerud` with its data and target as views.

    Args:
        return_views: Whether to return the list of views rather than a
            Bunch, as sklearn's ``return_X_y``. Default is False.

    Returns:
        A Bunch with ``views`` (exercise and physiological, each of shape
        (20, 3)), ``feature_names`` of each view and ``DESCR``; or the views.

    Examples:
        >>> from cca_zoo.datasets import load_linnerud
        >>> exercise, physiological = load_linnerud(return_views=True)
        >>> load_linnerud().feature_names[0]
        ['Chins', 'Situps', 'Jumps']
    """
    dataset = datasets.load_linnerud()
    views = [np.asarray(dataset.data), np.asarray(dataset.target)]
    if return_views:
        return views
    return Bunch(
        views=views,
        feature_names=[list(dataset.feature_names), list(dataset.target_names)],
        DESCR=dataset.DESCR,
    )


def load_breast_cancer(*, return_views: bool = False) -> Bunch | list[np.ndarray]:
    """The Wisconsin breast cancer features split into two 15-feature views.

    :func:`sklearn.datasets.load_breast_cancer`'s 30 features, the first half
    as one view and the second as the other.

    Args:
        return_views: Whether to return the list of views rather than a
            Bunch, as sklearn's ``return_X_y``. Default is False.

    Returns:
        A Bunch with ``views`` (each of shape (569, 15)), ``feature_names`` of
        each view and ``DESCR``; or the views.

    Examples:
        >>> from cca_zoo.datasets import load_breast_cancer
        >>> X1, X2 = load_breast_cancer(return_views=True)
        >>> X1.shape, X2.shape
        ((569, 15), (569, 15))
    """
    dataset = datasets.load_breast_cancer()
    data = np.asarray(dataset.data)
    names = list(dataset.feature_names)
    half = data.shape[1] // 2
    views = [data[:, :half], data[:, half:]]
    if return_views:
        return views
    return Bunch(
        views=views, feature_names=[names[:half], names[half:]], DESCR=dataset.DESCR
    )
