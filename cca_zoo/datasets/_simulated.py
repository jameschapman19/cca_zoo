"""Simulated multiview data from a linear latent variable model."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Literal, TypeVar, overload

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, validate_params

_T = TypeVar("_T", int, float)


def _per_view(value: _T | list[_T], n_views: int, name: str) -> list[_T]:
    """A scalar broadcast to every view, or a list checked to have one per view."""
    if isinstance(value, list):
        if len(value) != n_views:
            raise ValueError(
                f"{name} must be a scalar or a list of length {n_views}, "
                f"got {len(value)}."
            )
        return list(value)
    return [value] * n_views


@overload
def make_joint_data(
    n_samples: int = ...,
    n_features: int | list[int] = ...,
    n_views: int = ...,
    n_components: int = ...,
    signal_to_noise: float | list[float] = ...,
    random_state: int | np.random.RandomState | None = ...,
    return_latent: Literal[False] = ...,
) -> list[np.ndarray]: ...


@overload
def make_joint_data(
    n_samples: int = ...,
    n_features: int | list[int] = ...,
    n_views: int = ...,
    n_components: int = ...,
    signal_to_noise: float | list[float] = ...,
    random_state: int | np.random.RandomState | None = ...,
    *,
    return_latent: Literal[True],
) -> tuple[list[np.ndarray], np.ndarray]: ...


@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left"), list],
        "n_views": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "signal_to_noise": [Interval(Real, 0, None, closed="neither"), list],
        "random_state": ["random_state"],
        "return_latent": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def make_joint_data(
    n_samples: int = 100,
    n_features: int | list[int] = 10,
    n_views: int = 2,
    n_components: int = 1,
    signal_to_noise: float | list[float] = 1.0,
    random_state: int | np.random.RandomState | None = None,
    return_latent: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], np.ndarray]:
    """Generate multiview data from a linear latent variable model.

    Each view is ``x_i = z @ W_i.T + noise_i``, where ``z ~ N(0, I)`` is the
    shared latent variable of dimension ``n_components``, ``W_i ~ N(0, I)``
    is the view's loading matrix, and ``noise_i ~ N(0, I / signal_to_noise)``.
    For independent training and test sets, generate once and split with
    :func:`sklearn.model_selection.train_test_split`, which takes every view
    at once.

    Args:
        n_samples: Number of observations. Default is 100.
        n_features: Features per view: a single integer for every view or a
            list with one entry per view. Default is 10.
        n_views: Number of views. Default is 2.
        n_components: Dimension of the shared latent variable. Default is 1.
        signal_to_noise: Signal-to-noise ratio, a single float for every view
            or a list with one entry per view; higher means less noise.
            Default is 1.0.
        random_state: Seed or ``RandomState`` for reproducible output.
        return_latent: Also return the latent variable ``z``. Default is
            False.

    Returns:
        The list of views, each of shape (n_samples, n_features_i); with
        ``return_latent``, a tuple ``(views, z)`` with ``z`` of shape
        (n_samples, n_components).

    Raises:
        ValueError: If a per-view list has the wrong length.

    Examples:
        >>> from sklearn.model_selection import train_test_split
        >>> from cca_zoo.datasets import make_joint_data
        >>> views = make_joint_data(n_samples=200, n_components=2, random_state=0)
        >>> [v.shape for v in views]
        [(200, 10), (200, 10)]
        >>> X1_train, X1_test, X2_train, X2_test = train_test_split(
        ...     *views, random_state=0
        ... )
    """
    rng = check_random_state(random_state)
    features = _per_view(n_features, n_views, "n_features")
    snrs = _per_view(signal_to_noise, n_views, "signal_to_noise")
    latent = rng.standard_normal((n_samples, n_components))
    views = []
    for p, snr in zip(features, snrs):
        signal = latent @ rng.standard_normal((p, n_components)).T
        views.append(signal + rng.standard_normal(signal.shape) / np.sqrt(snr))
    if return_latent:
        return views, latent
    return views
