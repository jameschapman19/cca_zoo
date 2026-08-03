"""Tests for cca_zoo.datasets utilities.

The datasets module exposes JointData (a simulated multiview generator) and
two toy real-world loaders backed by scikit-learn's bundled datasets.
"""

from __future__ import annotations

import numpy as np

from cca_zoo.datasets import JointData, load_breast_cancer, load_linnerud

# ---------------------------------------------------------------------------
# JointData — simulated multiview data
# ---------------------------------------------------------------------------


class TestJointData:
    """Tests for JointData (linear latent variable data generator)."""

    def test_sample_returns_list_of_numpy_arrays(self) -> None:
        """sample() returns a list of numpy.ndarray, one per view."""
        gen = JointData(n_views=2, n_samples=50, n_features=[10, 8], random_state=0)
        views = gen.sample()
        assert isinstance(views, list)
        assert len(views) == 2
        for v in views:
            assert isinstance(v, np.ndarray)

    def test_sample_shapes_match_constructor_args(self) -> None:
        """Each view's shape matches the requested n_samples/n_features."""
        gen = JointData(n_views=2, n_samples=60, n_features=[10, 8], random_state=1)
        views = gen.sample()
        assert views[0].shape == (60, 10)
        assert views[1].shape == (60, 8)

    def test_sample_three_views(self) -> None:
        """JointData supports more than two views."""
        gen = JointData(
            n_views=3, n_samples=40, latent_dimensions=2, n_features=[6, 5, 4]
        )
        views = gen.sample()
        assert len(views) == 3
        for v, p in zip(views, [6, 5, 4]):
            assert v.shape == (40, p)

    def test_same_random_state_reproducible_across_instances(self) -> None:
        """Two freshly constructed generators with the same seed match."""
        gen1 = JointData(n_views=2, n_samples=20, n_features=5, random_state=0)
        gen2 = JointData(n_views=2, n_samples=20, n_features=5, random_state=0)
        for a, b in zip(gen1.sample(), gen2.sample()):
            np.testing.assert_array_equal(a, b)

    def test_repeated_sample_calls_draw_fresh_noise(self) -> None:
        """Calling sample() twice on the same instance gives different draws."""
        gen = JointData(n_views=2, n_samples=20, n_features=5, random_state=0)
        first = gen.sample()
        second = gen.sample()
        assert not np.array_equal(first[0], second[0])

    def test_call_is_alias_for_sample(self) -> None:
        """Calling the instance directly is equivalent to calling sample()."""
        gen = JointData(n_views=2, n_samples=20, n_features=5, random_state=2)
        views = gen()
        assert len(views) == 2
        for v in views:
            assert v.shape == (20, 5)


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


# ---------------------------------------------------------------------------
# Smoke test: __all__ lists are accessible
# ---------------------------------------------------------------------------


def test_datasets_all_attribute_exists() -> None:
    """cca_zoo.datasets defines __all__."""
    import cca_zoo.datasets as ds

    assert hasattr(ds, "__all__")
    assert isinstance(ds.__all__, list)
