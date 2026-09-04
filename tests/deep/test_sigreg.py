"""Tests for the SIGReg deep CCA model.

All tests are marked slow and require torch + lightning.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("lightning")

import torch.nn as nn
import torch.utils.data as data

from cca_zoo.deep._sigreg import (
    SIGReg,
    _invariance_loss,
    _sigreg_directions,
    _sigreg_loss,
)

# ---------------------------------------------------------------------------
# Helper: tiny multiview dataset and DataLoader
# ---------------------------------------------------------------------------


class _TinyMultiviewDataset(data.Dataset):
    """Minimal N-view dataset for testing."""

    def __init__(
        self, n: int = 40, p: int = 5, n_views: int = 2, seed: int = 0
    ) -> None:
        rng = np.random.default_rng(seed)
        self.views = [
            torch.from_numpy(rng.standard_normal((n, p)).astype(np.float32))
            for _ in range(n_views)
        ]

    def __len__(self) -> int:
        return len(self.views[0])

    def __getitem__(self, idx: int) -> dict:
        return {"views": [v[idx] for v in self.views]}


def _make_loader(
    n: int = 40, p: int = 5, n_views: int = 2, batch_size: int = 40
) -> data.DataLoader:
    dataset = _TinyMultiviewDataset(n=n, p=p, n_views=n_views)
    return data.DataLoader(dataset, batch_size=batch_size)


def _make_encoders(n_views: int = 2, p_in: int = 5, latent: int = 3) -> list[nn.Module]:
    return [nn.Linear(p_in, latent) for _ in range(n_views)]


# ---------------------------------------------------------------------------
# Helper functions — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sigreg_directions_unit_norm() -> None:
    """Sampled directions have unit norm and the requested shape."""
    directions = _sigreg_directions(
        latent_dimensions=8,
        num_directions=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert directions.shape == (8, 5)
    norms = directions.norm(dim=0)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


@pytest.mark.slow
def test_sigreg_loss_low_for_standard_gaussian_batch() -> None:
    """A large batch already ~N(0, I) yields a small SIGReg statistic."""
    torch.manual_seed(0)
    z = torch.randn(4096, 4)
    directions = _sigreg_directions(4, 16, z.device, z.dtype)
    nodes, weights = np.polynomial.hermite.hermgauss(17)
    quad_nodes = torch.as_tensor(np.sqrt(2) * nodes, dtype=torch.float32)
    quad_weights = torch.as_tensor(np.sqrt(2) * weights, dtype=torch.float32)
    stat = _sigreg_loss(z, directions, quad_nodes, quad_weights)
    assert float(stat) < 0.05


@pytest.mark.slow
def test_sigreg_loss_high_for_collapsed_batch() -> None:
    """A batch collapsed to a single point yields a large SIGReg statistic.

    This is the property SIGReg is designed to guarantee: the constant
    (zero-variance) solution is maximally far from an isotropic Gaussian.
    """
    z = torch.ones(256, 4) * 3.0
    directions = _sigreg_directions(4, 16, z.device, z.dtype)
    nodes, weights = np.polynomial.hermite.hermgauss(17)
    quad_nodes = torch.as_tensor(np.sqrt(2) * nodes, dtype=torch.float32)
    quad_weights = torch.as_tensor(np.sqrt(2) * weights, dtype=torch.float32)
    stat = _sigreg_loss(z, directions, quad_nodes, quad_weights)
    assert float(stat) > 1.0


@pytest.mark.slow
def test_invariance_loss_two_views_matches_mse() -> None:
    """With two views, the pairwise invariance loss is just their MSE."""
    z1 = torch.randn(16, 4)
    z2 = torch.randn(16, 4)
    expected = torch.nn.functional.mse_loss(z1, z2)
    assert torch.allclose(_invariance_loss([z1, z2]), expected)


@pytest.mark.slow
def test_invariance_loss_three_views_is_pairwise_mean() -> None:
    """With three views, the invariance loss averages all 3 pairwise MSEs."""
    z1 = torch.randn(16, 4)
    z2 = torch.randn(16, 4)
    z3 = torch.randn(16, 4)
    expected = (
        torch.nn.functional.mse_loss(z1, z2)
        + torch.nn.functional.mse_loss(z1, z3)
        + torch.nn.functional.mse_loss(z2, z3)
    ) / 3
    assert torch.allclose(_invariance_loss([z1, z2, z3]), expected)


# ---------------------------------------------------------------------------
# SIGReg model — end to end
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sigreg_loss_dict_keys() -> None:
    """SIGReg.loss returns the expected keys with scalar values."""
    latent = 3
    encoders = _make_encoders(2, 5, latent)
    model = SIGReg(latent_dimensions=latent, encoders=encoders)
    z1 = torch.randn(20, latent)
    z2 = torch.randn(20, latent)
    loss_dict = model.loss([z1, z2])
    assert set(loss_dict) == {"objective", "invariance", "sigreg"}
    for v in loss_dict.values():
        assert v.ndim == 0


@pytest.mark.slow
def test_sigreg_training_completes() -> None:
    """SIGReg trains for 2 epochs on tiny two-view data without error."""
    latent = 3
    encoders = _make_encoders(2, 5, latent)
    model = SIGReg(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    loader = _make_loader(n_views=2)
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)


@pytest.mark.slow
def test_sigreg_transform_output_shapes() -> None:
    """SIGReg transform returns arrays of shape (n_samples, latent_dimensions)."""
    latent = 3
    n, p = 40, 5
    encoders = _make_encoders(2, p, latent)
    model = SIGReg(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    loader = _make_loader(n=n, p=p, n_views=2)
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 2
    for arr in result:
        assert arr.shape == (n, latent)


@pytest.mark.slow
def test_sigreg_three_view_training() -> None:
    """SIGReg generalises to more than two views."""
    latent = 3
    n, p = 40, 5
    encoders = _make_encoders(3, p, latent)
    loader = _make_loader(n=n, p=p, n_views=3)
    model = SIGReg(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 3
    for arr in result:
        assert arr.shape == (n, latent)


@pytest.mark.slow
def test_sigreg_linear_encoders_reduce_correlation() -> None:
    """With linear encoders on correlated views, correlation rises with training."""
    torch.manual_seed(0)
    n, p, latent = 200, 6, 2
    rng = np.random.default_rng(1)
    shared = rng.standard_normal((n, latent)).astype(np.float32)
    noise1 = 0.1 * rng.standard_normal((n, p)).astype(np.float32)
    noise2 = 0.1 * rng.standard_normal((n, p)).astype(np.float32)
    x1 = shared @ rng.standard_normal((latent, p)).astype(np.float32) + noise1
    x2 = shared @ rng.standard_normal((latent, p)).astype(np.float32) + noise2

    class _Dataset(data.Dataset):
        def __len__(self) -> int:
            return n

        def __getitem__(self, idx: int) -> dict:
            return {
                "views": [
                    torch.from_numpy(x1[idx]),
                    torch.from_numpy(x2[idx]),
                ]
            }

    loader = data.DataLoader(_Dataset(), batch_size=n)
    encoders = [nn.Linear(p, latent, bias=False) for _ in range(2)]
    model = SIGReg(latent_dimensions=latent, encoders=encoders, max_epochs=50, lr=1e-2)
    trainer = lightning.pytorch.Trainer(
        max_epochs=50, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    score = model.score(loader)
    assert score.mean() > 0.5
