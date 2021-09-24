# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

import numpy as np
from sklearn.base import _is_pairwise
from sklearn.utils import _safe_indexing


def _safe_split(estimator, Xs, y, indices):
    """Create subset of dataset and properly handle kernels.
    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.
    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    Labels y will always be indexed only along the first axis.
    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.
    Xs : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.
    y : array-like, sparse matrix or iterable
        Targets to be indexed.
    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.
    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.
    y_subset : array-like, sparse matrix or list
        Indexed targets.
    """
    Xs_subset = []
    for X in Xs:
        if _is_pairwise(estimator):
            if not hasattr(X, "shape"):
                raise ValueError(
                    "Precomputed kernels or affinity matrices have "
                    "to be passed as arrays or sparse matrices."
                )
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            else:
                Xs_subset.append(X[np.ix_(indices, indices)])
        else:
            Xs_subset.append(_safe_indexing(X, indices))

    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None

    return Xs_subset, y_subset
