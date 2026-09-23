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
from cca_zoo.deep._data import MultiviewDataset
from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep._dgcca import DGCCA
from cca_zoo.deep._dmcca import DMCCA
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
# MultiviewDataset
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_multiview_dataset_batch_shape() -> None:
    """MultiviewDataset yields {"views": [...]} batches, not tuples."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((20, 5)).astype(np.float32)
    x2 = rng.standard_normal((20, 4)).astype(np.float32)
    loader = data.DataLoader(MultiviewDataset([x1, x2]), batch_size=20)
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert list(batch.keys()) == ["views"]
    assert len(batch["views"]) == 2
    assert batch["views"][0].shape == (20, 5)
    assert batch["views"][1].shape == (20, 4)


@pytest.mark.slow
def test_dcca_trains_with_multiview_dataset() -> None:
    """DCCA trains against a real DataLoader(MultiviewDataset(...)) end to end.

    Regression test: torch.utils.data.TensorDataset yields tuples, not the
    {"views": [...]} dict shape training_step/validation_step require, so a
    naive DataLoader(TensorDataset(...)) raises TypeError. This must keep
    passing against the documented docs/user-guide/deep.md example.
    """
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((40, 6)).astype(np.float32)
    x2 = rng.standard_normal((40, 5)).astype(np.float32)
    loader = data.DataLoader(MultiviewDataset([x1, x2]), batch_size=20, shuffle=True)

    encoders = [nn.Linear(6, 2), nn.Linear(5, 2)]
    model = DCCA(latent_dimensions=2, encoders=encoders, max_epochs=1)
    trainer = lightning.pytorch.Trainer(
        max_epochs=1, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 2
    for arr in result:
        assert arr.shape == (40, 2)


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


# ---------------------------------------------------------------------------
# DMCCA / DGCCA
# ---------------------------------------------------------------------------


def _make_three_view_loader(n: int = 20, p: int = 5) -> data.DataLoader:
    """Create a small DataLoader for three views."""

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

    return data.DataLoader(ThreeViewDataset(), batch_size=n)


@pytest.mark.slow
def test_dmcca_defaults_to_mcca_loss() -> None:
    """DMCCA uses MCCALoss regardless of the objective passed in."""
    latent = 2
    encoders = _make_encoders(5, latent)
    model = DMCCA(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    assert isinstance(model.objective, MCCALoss)


@pytest.mark.slow
def test_dmcca_three_view_training() -> None:
    """DMCCA trains on three-view data and transforms to the right shapes."""
    latent = 2
    n, p = 20, 5
    encoders = [nn.Linear(p, latent) for _ in range(3)]
    loader = _make_three_view_loader(n=n, p=p)
    model = DMCCA(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 3
    for arr in result:
        assert arr.shape == (n, latent)


@pytest.mark.slow
def test_dgcca_defaults_to_gcca_loss() -> None:
    """DGCCA uses GCCALoss regardless of the objective passed in."""
    latent = 2
    encoders = _make_encoders(5, latent)
    model = DGCCA(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    assert isinstance(model.objective, GCCALoss)


@pytest.mark.slow
def test_dgcca_three_view_training() -> None:
    """DGCCA trains on three-view data and transforms to the right shapes."""
    latent = 2
    n, p = 20, 5
    encoders = [nn.Linear(p, latent) for _ in range(3)]
    loader = _make_three_view_loader(n=n, p=p)
    model = DGCCA(latent_dimensions=latent, encoders=encoders, max_epochs=2)
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert len(result) == 3
    for arr in result:
        assert arr.shape == (n, latent)


# ---------------------------------------------------------------------------
# LeJEPA
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sigreg_small_for_gaussian_large_for_collapsed() -> None:
    """SIGReg (scaled by N) stays O(1) for isotropic Gaussians, grows otherwise."""
    from cca_zoo.deep._lejepa import _sigreg

    torch.manual_seed(0)
    gaussian = _sigreg(torch.randn(2048, 8), 0, 64, 17, 5.0)
    collapsed = _sigreg(torch.zeros(2048, 8), 0, 64, 17, 5.0)
    assert float(gaussian) < 2.0
    assert float(collapsed) > 100 * float(gaussian)


@pytest.mark.slow
def test_sigreg_slices_depend_on_seed() -> None:
    """Directions are reproducible for a seed and resampled across seeds."""
    from cca_zoo.deep._lejepa import _sigreg

    z = torch.randn(64, 4) * torch.tensor([3.0, 1.0, 0.2, 1.0])
    assert torch.equal(_sigreg(z, 1, 8, 17, 5.0), _sigreg(z, 1, 8, 17, 5.0))
    assert not torch.equal(_sigreg(z, 1, 8, 17, 5.0), _sigreg(z, 2, 8, 17, 5.0))


@pytest.mark.slow
def test_lejepa_loss_keys_and_identical_views() -> None:
    """Identical views have zero predictive loss; objective mixes both terms."""
    from cca_zoo.deep._lejepa import LeJEPA

    model = LeJEPA(latent_dimensions=2, encoders=_make_encoders(5, 2), lambd=0.3)
    z = torch.randn(16, 2)
    out = model.loss([z, z.clone()])
    assert float(out["sim_loss"]) == 0.0
    torch.testing.assert_close(out["objective"], 0.3 * out["sigreg"])


@pytest.mark.slow
def test_lejepa_three_view_training() -> None:
    """LeJEPA trains end-to-end with three views."""
    from cca_zoo.deep._lejepa import LeJEPA

    rng = np.random.default_rng(0)
    views = [rng.standard_normal((32, 5)).astype(np.float32) for _ in range(3)]
    loader = data.DataLoader(MultiviewDataset(views), batch_size=16)
    model = LeJEPA(
        latent_dimensions=2,
        encoders=[nn.Linear(5, 2) for _ in range(3)],
        num_slices=16,
    )
    trainer = lightning.pytorch.Trainer(
        max_epochs=2, enable_progress_bar=False, logger=False
    )
    trainer.fit(model, loader)
    result = model.transform(loader)
    assert [r.shape for r in result] == [(32, 2)] * 3
