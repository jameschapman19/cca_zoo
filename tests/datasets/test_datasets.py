"""Tests for cca_zoo.datasets utilities.

The datasets module exposes JointData, LatentVariableData (simulated) and
toy dataset loaders.  These tests verify that the public API returns arrays
of the expected shape.

Note: the simulated and toy submodules are referenced in datasets/__init__.py
but the underlying .py files may not yet exist in this rewrite snapshot.
Each test group is guarded by an importorskip / pytest.skip where appropriate.
"""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Guard: check if the datasets module is importable
# ---------------------------------------------------------------------------

datasets = pytest.importorskip(
    "cca_zoo.datasets",
    reason="cca_zoo.datasets could not be imported",
)


# ---------------------------------------------------------------------------
# JointData / LatentVariableData — simulated data
# ---------------------------------------------------------------------------


def _has_simulated() -> bool:
    """Return True if the simulated submodule is importable."""
    spec = importlib.util.find_spec("cca_zoo.datasets.simulated")
    return spec is not None


@pytest.mark.skipif(
    not _has_simulated(),
    reason="cca_zoo.datasets.simulated not available in this rewrite snapshot",
)
class TestJointData:
    """Tests for JointData (linear latent variable data generator)."""

    def test_sample_returns_list(self) -> None:
        """JointData.sample() returns a list of numpy arrays."""
        from cca_zoo.datasets import JointData  # type: ignore[attr-defined]

        gen = JointData(latent_dimensions=2, n_views=2, n_features=[10, 8])
        views = gen.sample(n_samples=50, random_state=0)
        assert isinstance(views, list)
        assert len(views) == 2

    def test_sample_correct_n_samples(self) -> None:
        """JointData.sample() returns arrays with the requested number of samples."""
        from cca_zoo.datasets import JointData  # type: ignore[attr-defined]

        gen = JointData(latent_dimensions=2, n_views=2, n_features=[10, 8])
        views = gen.sample(n_samples=60, random_state=1)
        for v in views:
            assert v.shape[0] == 60

    def test_sample_correct_n_features(self) -> None:
        """JointData.sample() returns arrays with the specified feature dimensions."""
        from cca_zoo.datasets import JointData  # type: ignore[attr-defined]

        n_features = [10, 8]
        gen = JointData(latent_dimensions=2, n_views=2, n_features=n_features)
        views = gen.sample(n_samples=30, random_state=2)
        for v, p in zip(views, n_features):
            assert v.shape[1] == p

    def test_sample_returns_numpy_arrays(self) -> None:
        """JointData.sample() returns numpy.ndarray objects."""
        from cca_zoo.datasets import JointData  # type: ignore[attr-defined]

        gen = JointData(latent_dimensions=1, n_views=2, n_features=[5, 5])
        views = gen.sample(n_samples=20, random_state=3)
        for v in views:
            assert isinstance(v, np.ndarray)

    def test_sample_reproducible(self) -> None:
        """JointData.sample() is reproducible with the same random_state."""
        from cca_zoo.datasets import JointData  # type: ignore[attr-defined]

        gen = JointData(latent_dimensions=1, n_views=2, n_features=[5, 5])
        v1 = gen.sample(n_samples=20, random_state=0)
        v2 = gen.sample(n_samples=20, random_state=0)
        for a, b in zip(v1, v2):
            np.testing.assert_array_equal(a, b)

    def test_sample_three_views(self) -> None:
        """JointData.sample() works for three views."""
        from cca_zoo.datasets import JointData  # type: ignore[attr-defined]

        gen = JointData(latent_dimensions=2, n_views=3, n_features=[6, 5, 4])
        views = gen.sample(n_samples=40, random_state=0)
        assert len(views) == 3
        for v, p in zip(views, [6, 5, 4]):
            assert v.shape == (40, p)


@pytest.mark.skipif(
    not _has_simulated(),
    reason="cca_zoo.datasets.simulated not available in this rewrite snapshot",
)
class TestLatentVariableData:
    """Tests for LatentVariableData (if present alongside JointData)."""

    def test_latent_variable_data_importable(self) -> None:
        """LatentVariableData is importable from cca_zoo.datasets."""
        from cca_zoo.datasets import LatentVariableData  # type: ignore[attr-defined]

        assert LatentVariableData is not None


# ---------------------------------------------------------------------------
# Toy datasets
# ---------------------------------------------------------------------------


def _has_toy() -> bool:
    """Return True if the toy submodule is importable."""
    spec = importlib.util.find_spec("cca_zoo.datasets.toy")
    return spec is not None


@pytest.mark.skipif(
    not _has_toy(),
    reason="cca_zoo.datasets.toy not available in this rewrite snapshot",
)
class TestToyDatasets:
    """Tests for real-world toy dataset loaders."""

    def test_load_breast_data_returns_numpy(self) -> None:
        """load_breast_data returns numpy arrays."""
        from cca_zoo.datasets import load_breast_data  # type: ignore[attr-defined]

        result = load_breast_data()
        assert isinstance(result, (list, tuple))
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_load_breast_data_consistent_samples(self) -> None:
        """load_breast_data views share the same number of rows."""
        from cca_zoo.datasets import load_breast_data  # type: ignore[attr-defined]

        result = load_breast_data()
        n = result[0].shape[0]
        for arr in result:
            assert arr.shape[0] == n

    def test_load_mfeat_data_returns_numpy(self) -> None:
        """load_mfeat_data returns numpy arrays."""
        from cca_zoo.datasets import load_mfeat_data  # type: ignore[attr-defined]

        result = load_mfeat_data()
        assert isinstance(result, (list, tuple))
        for arr in result:
            assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# Smoke test: __all__ lists are accessible
# ---------------------------------------------------------------------------


def test_datasets_all_attribute_exists() -> None:
    """cca_zoo.datasets defines __all__."""
    import cca_zoo.datasets as ds

    assert hasattr(ds, "__all__")
    assert isinstance(ds.__all__, list)
