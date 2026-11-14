"""Check an embedding for coordinates that are functions of earlier ones."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "scores": ["array-like"],
        "n_splits": [Interval(Integral, 2, None, closed="left")],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def nonlinear_redundancy(
    scores: ArrayLike,
    n_splits: int = 3,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    r"""Held-out $R^2$ of each coordinate from a flexible function of the earlier ones.

    A loss that ranks functions by cross-view predictability, such as the
    Eckart-Young loss with flexible encoders, only makes the coordinates
    uncorrelated. A function of a very stable factor, such as its square, is
    uncorrelated with the factor yet fills a coordinate that a less stable
    factor needed. Such a coordinate is predicted well by a nonlinear
    regression on the coordinates before it. Coordinate $j$ is regressed on
    coordinates $0, \dots, j - 1$ by gradient-boosted trees, with the $R^2$
    taken over held-out folds; a value near 1 means coordinate $j$ is almost a
    function of the earlier ones. The fit costs $d - 1$ boosted regressions per
    fold.

    Args:
        scores: Embedding of one view, shape (n_samples, n_components), with
            the coordinates in the order they were learned.
        n_splits: Number of cross-validation folds. Default is 3.
        random_state: Seed for the folds. Default is None.

    Returns:
        Held-out $R^2$ per coordinate, shape (n_components,), clipped at 0. The
        first coordinate has nothing before it and is 0.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.model_selection import nonlinear_redundancy
        >>> z = np.random.default_rng(0).standard_normal((2000, 2))
        >>> embedding = np.column_stack([z[:, 0], z[:, 0] ** 2 - 1, z[:, 1]])
        >>> nonlinear_redundancy(embedding, random_state=0) > 0.5
        array([False,  True, False])
    """
    x = check_array(scores, dtype=np.float64)
    folds = KFold(n_splits, shuffle=True, random_state=random_state)
    redundancy = np.zeros(x.shape[1])
    for j in range(1, x.shape[1]):
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=15, min_samples_leaf=20, random_state=0
        )
        prediction = cross_val_predict(model, x[:, :j], x[:, j], cv=folds)
        residual = np.sum((x[:, j] - prediction) ** 2)
        redundancy[j] = max(
            0.0, 1.0 - residual / np.sum((x[:, j] - x[:, j].mean()) ** 2)
        )
    return redundancy
