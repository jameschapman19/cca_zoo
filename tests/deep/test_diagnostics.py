"""The H/Z diagnostics. Slow; need torch and lightning."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import lightning.pytorch as pl
import torch.nn as nn
from torch.utils.data import DataLoader

from cca_zoo.deep import DCCAEY, MultiviewDataset
from cca_zoo.deep._diagnostics import HZDiagnostics, representation_stats


def test_shared_component_dominates_stats() -> None:
    torch.manual_seed(0)
    noise = 0.01 * torch.randn(512, 16)
    stats = representation_stats(100.0 * torch.ones(1, 16) + noise)
    assert stats["shared"] > 0.999
    assert 1 <= stats["erank"] <= 16


def test_isotropic_features_have_full_erank() -> None:
    torch.manual_seed(0)
    stats = representation_stats(torch.randn(4096, 8))
    assert stats["shared"] < 0.01
    assert stats["erank"] == pytest.approx(8, rel=0.02)


def test_flattens_token_dimensions() -> None:
    assert representation_stats(torch.randn(4, 5, 8))["erank"] <= 8


def test_callback_logs_h_and_z() -> None:
    torch.manual_seed(0)
    views = [torch.randn(256, 10), torch.randn(256, 6)]
    model = DCCAEY(
        n_components=3,
        encoders=[
            nn.Sequential(nn.Sequential(nn.Linear(d, 8), nn.ReLU()), nn.Linear(8, 3))
            for d in (10, 6)
        ],
    )
    callback = HZDiagnostics(lambda m, v: m.encoders[v][0], every_n_steps=1)
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, DataLoader(MultiviewDataset(views), batch_size=64))
    for name in ("H0", "H1", "Z0", "Z1"):
        assert f"diag/{name}/erank" in trainer.callback_metrics
