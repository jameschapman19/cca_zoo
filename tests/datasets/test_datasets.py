"""Tests for cca_zoo.datasets utilities.

The datasets module exposes make_joint_data (a simulated multiview generator)
and two toy real-world loaders backed by scikit-learn's bundled datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.datasets import load_breast_cancer, load_linnerud, make_joint_data

# ---------------------------------------------------------------------------
# make_joint_data — simulated multiview data
# ---------------------------------------------------------------------------


class TestMakeJointData:
    """Tests for make_joint_data (linear latent variable data generator)."""

    def test_returns_one_array_per_view_with_requested_shapes(self) -> None:
        """Each view is (n_samples, n_features_i), for any number of views."""
        views = make_joint_data(n_samples=40, n_features=[6, 5, 4], n_views=3)
        assert [v.shape for v in views] == [(40, 6), (40, 5), (40, 4)]

    def test_scalar_n_features_broadcasts(self) -> None:
        """A single n_features applies to every view."""
        views = make_joint_data(n_samples=20, n_features=5, n_views=2)
        assert [v.shape for v in views] == [(20, 5), (20, 5)]

    def test_same_random_state_reproducible(self) -> None:
        """The same seed gives the same data."""
        for a, b in zip(
            make_joint_data(random_state=0), make_joint_data(random_state=0)
        ):
            np.testing.assert_array_equal(a, b)

    def test_return_latent_explains_the_views(self) -> None:
        """With return_latent, z explains most of every view at high SNR."""
        views, z = make_joint_data(
            n_samples=500,
            n_components=2,
            signal_to_noise=100.0,
            random_state=0,
            return_latent=True,
        )
        assert z.shape == (500, 2)
        for v in views:
            residual = v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
            assert residual.var() < 0.05 * v.var()

    def test_wrong_length_per_view_list_raises(self) -> None:
        """A per-view list must have one entry per view."""
        with pytest.raises(ValueError, match="n_features"):
            make_joint_data(n_features=[5, 5, 5], n_views=2)


# ---------------------------------------------------------------------------
# Toy datasets
# ---------------------------------------------------------------------------


def test_load_breast_cancer_returns_two_equal_views() -> None:
    """load_breast_cancer splits the 30 features into two 15-feature views."""
    x1, x2 = load_breast_cancer()
    assert x1.shape == (569, 15)
    assert x2.shape == (569, 15)


def test_load_linnerud_returns_expected_shapes() -> None:
    """load_linnerud returns the exercise/physiological views."""
    x1, x2 = load_linnerud()
    assert x1.shape == (20, 3)
    assert x2.shape == (20, 3)
