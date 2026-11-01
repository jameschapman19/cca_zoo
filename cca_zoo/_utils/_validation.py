"""Input validation utilities for multiview data."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse.csgraph import connected_components
from sklearn.utils._array_api import get_namespace
from sklearn.utils.validation import check_array

_T = TypeVar("_T")


def validate_views(
    views: list[ArrayLike],
    min_views: int = 2,
    ensure_all_finite: bool | str = True,
    ensure_min_samples: int = 1,
    dtype: type | list[type] = np.float64,
) -> list[np.ndarray]:
    """Validate and convert multiview data to a list of 2-D numpy arrays.

    Args:
        views: List of array-like objects, each of shape (n_samples, n_features_i).
        min_views: Minimum number of views required. Default is 2.
        ensure_all_finite: Whether to reject NaN/inf entries. Default is
            ``True``. ``"allow-nan"`` rejects only inf, for callers that handle
            missing values themselves (e.g. a tree encoder); callers that
            handle every non-finite value (e.g. an imputer) pass ``False``.
        ensure_min_samples: Minimum number of samples. Default is 1.
        dtype: The dtype to convert to, or a list of dtypes to keep, as in
            :func:`~sklearn.utils.check_array`. Views of mixed precision are
            all converted to the first. Default is float64.

    Returns:
        List of validated 2-D arrays, each of shape (n_samples, n_features_i), in
        the views' namespace.

    Raises:
        ValueError: If fewer than ``min_views`` views are provided.
        ValueError: If views have inconsistent numbers of samples.
    """
    if len(views) < min_views:
        raise ValueError(f"At least {min_views} views are required, got {len(views)}.")
    # The dtypes in the views' own namespace, for Array API inputs.
    xp, _ = get_namespace(*views)
    requested = dtype if isinstance(dtype, list) else [dtype]
    dtypes = [getattr(xp, np.dtype(t).name) for t in requested]
    processed = [
        check_array(
            v,
            ensure_2d=True,
            allow_nd=False,
            dtype=dtypes,
            ensure_all_finite=ensure_all_finite,
            ensure_min_samples=ensure_min_samples,
        )
        for v in views
    ]
    if len({v.dtype for v in processed}) > 1:
        processed = [xp.astype(v, dtypes[0]) for v in processed]
    n_samples = processed[0].shape[0]
    if not all(v.shape[0] == n_samples for v in processed):
        raise ValueError(
            "All views must have the same number of samples. "
            f"Got shapes: {[v.shape for v in processed]}."
        )
    return processed


def perview_parameter(
    name: str,
    value: _T | list[_T] | None,
    default: _T,
    n_views: int,
) -> list[_T]:
    """Broadcast a scalar or per-view parameter to a list of length n_views.

    Args:
        name: Parameter name (used in error messages).
        value: Scalar value, list of values, or None (uses default).
        default: Default value used when value is None.
        n_views: Number of views.

    Returns:
        List with exactly ``n_views`` elements.

    Raises:
        ValueError: If value is a list with incorrect length.
    """
    if value is None:
        return [default] * n_views
    if isinstance(value, list):
        if len(value) != n_views:
            raise ValueError(
                f"Parameter '{name}' must be a scalar or a list of length "
                f"{n_views}, got length {len(value)}."
            )
        return value
    # scalar broadcast (covers int, float, str, etc.)
    return [value] * n_views


def observed_rows(views: list[np.ndarray], n_components: int, model: str) -> np.ndarray:
    """Which samples each view is observed in, for views with missing samples.

    A sample lacks a view when its whole row is ``NaN``.

    Args:
        views: Arrays of shape (n_samples, n_features_i).
        n_components: Latent dimensions to be fitted.
        model: Name of the model, for the error messages.

    Returns:
        Boolean array of shape (n_samples, n_views).

    Raises:
        ValueError: If a row is partly missing, a sample is in no view, a view
            is observed in fewer than 2 samples, or the views are not linked
            by more than ``n_components`` shared samples, directly or through
            other views: without that their scores are not tied together.
    """
    missing = [np.isnan(v) for v in views]
    for m in missing:
        partly = m.any(axis=1) & ~m.all(axis=1)
        if partly.any():
            raise ValueError(
                "A view's row must be wholly observed or wholly NaN, but rows "
                f"{np.flatnonzero(partly)[:5]} are partly missing. Impute missing "
                "entries within an observed view first."
            )
    observed = np.stack([~m.any(axis=1) for m in missing], axis=1)
    if not observed.any(axis=1).all():
        raise ValueError(
            f"{model} needs every sample to be observed in at least one view; "
            f"samples {np.flatnonzero(~observed.any(axis=1))} have none."
        )
    if (observed.sum(axis=0) < 2).any():
        raise ValueError("Every view must be observed in at least 2 samples.")
    shared = observed.T.astype(int) @ observed.astype(int)
    if connected_components(shared > n_components, directed=False)[0] > 1:
        raise ValueError(
            "The views are not linked: they must be joined, directly or "
            f"through other views, by more than n_components={n_components} "
            f"shared samples. Samples shared by each pair of views:\n{shared}"
        )
    return observed
