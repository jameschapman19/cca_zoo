"""The shared input and per-view parameter validation."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._utils._validation import perview_parameter, validate_views


def test_validate_views_returns_float_arrays() -> None:
    """Nested lists become float64 arrays of the same shapes."""
    views = validate_views([[[1, 2], [3, 4]], [[5], [6]]])
    assert [(v.dtype, v.shape) for v in views] == [
        (np.float64, (2, 2)),
        (np.float64, (2, 1)),
    ]


@pytest.mark.parametrize(
    ("views", "kwargs", "match"),
    [
        ([np.ones((3, 2))], {}, "At least 2 views"),
        ([np.ones((3, 2)), np.ones((4, 2))], {}, "same number of samples"),
        ([np.ones((3, 2)), np.full((3, 2), np.nan)], {}, "NaN"),
        ([np.ones((1, 2)), np.ones((1, 2))], {"ensure_min_samples": 2}, "minimum"),
    ],
)
def test_validate_views_rejects(
    views: list[np.ndarray], kwargs: dict[str, int], match: str
) -> None:
    """Too few views, mismatched samples, NaN and too few samples raise."""
    with pytest.raises(ValueError, match=match):
        validate_views(views, **kwargs)


def test_validate_views_lets_nan_through_for_imputers() -> None:
    """ensure_all_finite=False keeps NaN, for a per-view imputer."""
    views = validate_views(
        [np.ones((3, 2)), np.full((3, 2), np.nan)], ensure_all_finite=False
    )
    assert np.isnan(views[1]).all()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, [0.5, 0.5, 0.5]), (None, [0.1, 0.1, 0.1]), ([1, 2, 3], [1, 2, 3])],
)
def test_perview_parameter_broadcasts(
    value: float | list[int] | None, expected: list[float]
) -> None:
    """A scalar is broadcast, None gives the default and a list passes through."""
    assert perview_parameter("shrinkage", value, 0.1, 3) == expected


def test_perview_parameter_needs_one_value_per_view() -> None:
    """A list of the wrong length names the parameter and the length expected."""
    with pytest.raises(ValueError, match="'shrinkage' must be a scalar or a list of length 3"):
        perview_parameter("shrinkage", [0.1, 0.2], 0.0, 3)
