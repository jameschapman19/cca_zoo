"""Tests for deep CCA models.

All tests are marked slow and require torch + lightning.
These tests import directly from the deep submodules rather than from
cca_zoo.deep, since the package __init__.py references discriminative/
generative sub-packages that are not yet present in the v3 rewrite tree.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("lightning")

import torch.nn as nn
import torch.utils.data as data

# ---------------------------------------------------------------------------
# Import the available deep classes directly
# ---------------------------------------------------------------------------
from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep.objectives import (
    CCALoss,
    GCCALoss,
    MCCALoss,
    TCCALoss,
)

# ---------------------------------------------------------------------------
# Helper: tiny dataset and DataLoader
# ---------------------------------------------------------------------------


class _TinyViewDataset(data.Dataset):
    """Minimal two-view dataset for testing."""

    def __init__(
        self,
        n: int = 20,
        p1: int = 5,
        p2: int = 5,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.x1 = torch.from_numpy(rng.standard_normal((n, p1)).astype(np.float32))
        self.x2 = torch.from_numpy(rng.standard_normal((n, p2)).astype(np.float32))

    def __len__(self) -> int:
        return len(self.x1)

    def __getitem__(self, idx: int) -> dict:
        return {"views": [self.x1[idx], self.x2[idx]]}


def _make_loader(n: int = 20, p: int = 5, batch_size: int = 20) -> data.DataLoader:
    """Create a small DataLoader for two views."""
    dataset = _TinyViewDataset(n=n, p1=p, p2=p)
    return data.DataLoader(dataset, batch_size=batch_size)


def _make_encoders(p_in: int = 5, latent: int = 2) -> list[nn.Module]:
    """Create two simple linear encoders."""
    return [nn.Linear(p_in, latent), nn.Linear(p_in, latent)]


# ---------------------------------------------------------------------------
# BaseDeep / DCCA — basic training and transform
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dcca_training_completes() -> None:
    """DCCA trains for 2 epochs on tiny data without error."""
    latent = 2
    encoders = _make_encoders(5, latent)
    model = DCCA(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    loader = _make_loader()
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)


@pytest.mark.slow
def test_dcca_transform_output_shapes() -> None:
    """DCCA transform returns arrays of shape (n_samples, latent_dimensions)."""
    latent = 2
    n = 20
    p = 5
    encoders = _make_encoders(p, latent)
    model = DCCA(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    loader = _make_loader(n=n, p=p)
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 2
    for arr in result:
        assert arr.shape == (n, latent)


@pytest.mark.slow
def test_dcca_score_shape() -> None:
    """DCCA score returns array of shape (latent_dimensions,)."""
    latent = 2
    encoders = _make_encoders(5, latent)
    model = DCCA(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    loader = _make_loader()
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    s = model.score(loader)
    assert s.shape == (latent,)


@pytest.mark.slow
def test_dcca_with_mcca_objective() -> None:
    """DCCA works when given a custom MCCALoss objective."""
    latent = 2
    encoders = _make_encoders(5, latent)
    model = DCCA(
        latent_dimensions=latent,
        encoders=encoders,
        objective=MCCALoss(eps=1e-4),
        max_epochs=2,
    )
    loader = _make_loader()
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 2


@pytest.mark.slow
def test_dcca_with_gcca_objective() -> None:
    """DCCA works when given a custom GCCALoss objective."""
    latent = 2
    encoders = _make_encoders(5, latent)
    model = DCCA(
        latent_dimensions=latent,
        encoders=encoders,
        objective=GCCALoss(eps=1e-4),
        max_epochs=2,
    )
    loader = _make_loader()
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Objectives — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cca_loss_scalar() -> None:
    """CCALoss returns a scalar tensor."""
    loss_fn = CCALoss(eps=1e-4)
    z1 = torch.randn(16, 4)
    z2 = torch.randn(16, 4)
    loss = loss_fn([z1, z2])
    assert loss.ndim == 0


@pytest.mark.slow
def test_cca_loss_negative() -> None:
    """CCALoss is non-positive (minimising it maximises correlation)."""
    loss_fn = CCALoss(eps=1e-4)
    z1 = torch.randn(16, 4)
    z2 = torch.randn(16, 4)
    loss = loss_fn([z1, z2])
    assert float(loss) <= 0.0


@pytest.mark.slow
def test_cca_loss_wrong_n_views_raises() -> None:
    """CCALoss raises ValueError if given != 2 views."""
    loss_fn = CCALoss()
    with pytest.raises(ValueError, match="exactly 2"):
        loss_fn([torch.randn(8, 4), torch.randn(8, 4), torch.randn(8, 4)])


@pytest.mark.slow
def test_mcca_loss_scalar() -> None:
    """MCCALoss returns a scalar tensor for 3 views."""
    loss_fn = MCCALoss(eps=1e-4)
    views = [torch.randn(16, 4) for _ in range(3)]
    loss = loss_fn(views)
    assert loss.ndim == 0


@pytest.mark.slow
def test_gcca_loss_scalar() -> None:
    """GCCALoss returns a scalar tensor for 3 views."""
    loss_fn = GCCALoss(eps=1e-4)
    views = [torch.randn(16, 4) for _ in range(3)]
    loss = loss_fn(views)
    assert loss.ndim == 0


@pytest.mark.slow
def test_tcca_loss_scalar() -> None:
    """TCCALoss returns a scalar tensor for 3 views."""
    loss_fn = TCCALoss(eps=1e-4)
    views = [torch.randn(16, 4) for _ in range(3)]
    loss = loss_fn(views)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# BaseDeep — forward method
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_base_deep_forward_output_shapes() -> None:
    """BaseDeep forward method returns latent representations of correct shape."""
    latent = 3
    encoders = _make_encoders(5, latent)
    model = DCCA(latent_dimensions=latent, encoders=encoders)
    x1 = torch.randn(8, 5)
    x2 = torch.randn(8, 5)
    result = model([x1, x2])
    assert len(result) == 2
    for r in result:
        assert r.shape == (8, latent)


# ---------------------------------------------------------------------------
# DCCA with three-view data
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dcca_three_view_training() -> None:
    """DCCA with three encoders trains on three-view data."""
    latent = 2
    n, p = 20, 5
    encoders = [nn.Linear(p, latent) for _ in range(3)]

    class ThreeViewDataset(data.Dataset):
        def __init__(self) -> None:
            rng = np.random.default_rng(0)
            self.views = [
                torch.from_numpy(rng.standard_normal((n, p)).astype(np.float32))
                for _ in range(3)
            ]

        def __len__(self) -> int:
            return n

        def __getitem__(self, idx: int) -> dict:
            return {"views": [v[idx] for v in self.views]}

    loader = data.DataLoader(ThreeViewDataset(), batch_size=n)
    model = DCCA(
        latent_dimensions=latent,
        encoders=encoders,
        objective=MCCALoss(eps=1e-4),
        max_epochs=2,
    )
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 3
    for arr in result:
        assert arr.shape == (n, latent)
