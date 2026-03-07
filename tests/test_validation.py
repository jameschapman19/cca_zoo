"""Tests for cca_zoo._utils._validation."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._utils._validation import perview_parameter, validate_views

# ---------------------------------------------------------------------------
# validate_views
# ---------------------------------------------------------------------------


def test_validate_views_returns_list_of_ndarrays() -> None:
    """validate_views converts array-like inputs to a list of numpy arrays."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((20, 5))
    x2 = rng.standard_normal((20, 4))
    result = validate_views([x1, x2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(v, np.ndarray) for v in result)


def test_validate_views_accepts_list_input() -> None:
    """validate_views correctly handles plain Python lists as views."""
    x1 = [[1.0, 2.0], [3.0, 4.0]]
    x2 = [[5.0, 6.0], [7.0, 8.0]]
    result = validate_views([x1, x2])
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)


def test_validate_views_raises_on_one_view() -> None:
    """validate_views raises ValueError with only one view."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 5))
    with pytest.raises(ValueError, match="At least 2 views"):
        validate_views([x])


def test_validate_views_raises_on_zero_views() -> None:
    """validate_views raises ValueError with an empty list."""
    with pytest.raises(ValueError, match="At least 2 views"):
        validate_views([])


def test_validate_views_raises_on_inconsistent_samples() -> None:
    """validate_views raises ValueError when row counts differ across views."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((10, 5))
    x2 = rng.standard_normal((15, 5))
    with pytest.raises(ValueError, match="same number of samples"):
        validate_views([x1, x2])


def test_validate_views_output_shapes_preserved() -> None:
    """validate_views preserves view shapes exactly."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((30, 7))
    x2 = rng.standard_normal((30, 4))
    x3 = rng.standard_normal((30, 9))
    result = validate_views([x1, x2, x3])
    assert result[0].shape == (30, 7)
    assert result[1].shape == (30, 4)
    assert result[2].shape == (30, 9)


def test_validate_views_three_views_ok() -> None:
    """validate_views accepts three views without error."""
    rng = np.random.default_rng(0)
    views = [rng.standard_normal((20, i + 3)) for i in range(3)]
    result = validate_views(views)
    assert len(result) == 3


def test_validate_views_custom_min_views() -> None:
    """validate_views respects a custom min_views argument."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 5))
    result = validate_views([x], min_views=1)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# perview_parameter — scalar broadcast
# ---------------------------------------------------------------------------


def test_perview_parameter_scalar_broadcast() -> None:
    """A scalar value is broadcast to all n_views views."""
    result = perview_parameter("c", 0.5, 0.0, 3)
    assert result == [0.5, 0.5, 0.5]


def test_perview_parameter_int_broadcast() -> None:
    """An integer value is broadcast to all views."""
    result = perview_parameter("span", 5, 1, 2)
    assert result == [5, 5]
    assert all(v == 5 for v in result)


# ---------------------------------------------------------------------------
# perview_parameter — list passthrough
# ---------------------------------------------------------------------------


def test_perview_parameter_list_passthrough() -> None:
    """A list of the correct length is passed through as floats."""
    result = perview_parameter("c", [0.1, 0.2, 0.3], 0.0, 3)
    assert result == [0.1, 0.2, 0.3]


def test_perview_parameter_list_passthrough_ints() -> None:
    """A list of integers is passed through unchanged."""
    result = perview_parameter("c", [1, 2], 0, 2)
    assert result == [1, 2]


# ---------------------------------------------------------------------------
# perview_parameter — None returns defaults
# ---------------------------------------------------------------------------


def test_perview_parameter_none_returns_default() -> None:
    """None value returns a list of the default repeated n_views times."""
    result = perview_parameter("c", None, 0.1, 4)
    assert result == [0.1, 0.1, 0.1, 0.1]


def test_perview_parameter_none_default_zero() -> None:
    """None value with default 0.0 returns all zeros."""
    result = perview_parameter("alpha", None, 0.0, 2)
    assert result == [0.0, 0.0]


# ---------------------------------------------------------------------------
# perview_parameter — wrong length raises ValueError
# ---------------------------------------------------------------------------


def test_perview_parameter_wrong_length_raises() -> None:
    """A list of wrong length raises ValueError with informative message."""
    with pytest.raises(ValueError, match="length 3"):
        perview_parameter("c", [0.1, 0.2], 0.0, 3)


def test_perview_parameter_too_long_raises() -> None:
    """A list that is too long raises ValueError."""
    with pytest.raises(ValueError, match="length 2"):
        perview_parameter("c", [0.1, 0.2, 0.3], 0.0, 2)


# ---------------------------------------------------------------------------
# perview_parameter — edge cases
# ---------------------------------------------------------------------------


def test_perview_parameter_single_view_with_list() -> None:
    """A one-element list matches n_views=1."""
    result = perview_parameter("c", [0.7], 0.0, 1)
    assert result == [0.7]


def test_perview_parameter_two_views_two_values() -> None:
    """Exactly matching lengths work for two views."""
    result = perview_parameter("c", [0.1, 0.9], 0.0, 2)
    assert result == [0.1, 0.9]
