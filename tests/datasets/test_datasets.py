"""The simulated and bundled datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cca_zoo.datasets import load_breast_cancer, load_linnerud, make_joint_data


def test_make_joint_data_shapes() -> None:
    """One array per view, with n_features per view or broadcast from a scalar."""
    assert [v.shape for v in make_joint_data(40, [6, 5, 4], n_views=3)] == [
        (40, 6),
        (40, 5),
        (40, 4),
    ]
    assert [v.shape for v in make_joint_data(20, 5, n_views=2)] == [(20, 5), (20, 5)]
    with pytest.raises(ValueError, match="n_features"):
        make_joint_data(n_features=[5, 5, 5], n_views=2)


def test_latent_explains_the_views() -> None:
    """At high signal to noise, the returned latent explains every view."""
    views, z = make_joint_data(
        500, n_components=2, signal_to_noise=100.0, random_state=0, return_latent=True
    )
    for v in views:
        residual = v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
        assert residual.var() < 0.05 * v.var()


@pytest.mark.parametrize(
    ("loader", "shapes"),
    [(load_breast_cancer, [(569, 15)] * 2), (load_linnerud, [(20, 3)] * 2)],
)
def test_bundled_datasets(loader: Any, shapes: list[tuple[int, int]]) -> None:
    """The bundled sklearn datasets load as a Bunch of views, or the views alone."""
    bunch = loader()
    assert [v.shape for v in bunch.views] == shapes
    assert [len(names) for names in bunch.feature_names] == [s[1] for s in shapes]
    assert [v.shape for v in loader(return_views=True)] == shapes
