"""Input validation utilities for multiview data."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_array

_T = TypeVar("_T")


def validate_views(
    views: list[ArrayLike],
    min_views: int = 2,
) -> list[np.ndarray]:
    """Validate and convert multiview data to a list of 2-D numpy arrays.

    Args:
        views: List of array-like objects, each of shape (n_samples, n_features_i).
        min_views: Minimum number of views required. Default is 2.

    Returns:
        List of validated numpy arrays, each of shape (n_samples, n_features_i).

    Raises:
        ValueError: If fewer than ``min_views`` views are provided.
        ValueError: If views have inconsistent numbers of samples.
    """
    if len(views) < min_views:
        raise ValueError(f"At least {min_views} views are required, got {len(views)}.")
    processed = [
        check_array(v, ensure_2d=True, allow_nd=False, dtype="numeric") for v in views
    ]
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
    return [value] * n_views  # type: ignore[return-value]
