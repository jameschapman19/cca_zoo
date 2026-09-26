"""Tests for the deep models. All are marked slow and need torch and lightning."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("lightning")

import torch.nn as nn
from torch.utils.data import DataLoader

from cca_zoo.deep import (
    DCCA,
    DCCAE,
    DCCAEY,
    DCCANOI,
    DCCASDL,
    DGCCA,
    DMCCA,
    DTCCA,
    DVCCA,
    BarlowTwins,
    BaseDeep,
    MultiviewDataset,
    SplitAE,
    VICReg,
)
from cca_zoo.deep.objectives import CCALoss, GCCALoss, MCCALoss, TCCALoss
from cca_zoo.metrics import pairwise_correlations

pytestmark = pytest.mark.slow
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

K = 2
P = (6, 5)


def _views(n: int = 64, seed: int = 0, n_views: int = 2) -> list[np.ndarray]:
    """Views sharing a two-dimensional latent signal."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, K))
    widths = (*P, 4)[:n_views]
    return [
        (z @ rng.standard_normal((K, p)) + 0.3 * rng.standard_normal((n, p))).astype(
            np.float32
        )
        for p in widths
    ]


def _loader(views: list[np.ndarray], shuffle: bool = False) -> DataLoader:
    return DataLoader(MultiviewDataset(views), batch_size=32, shuffle=shuffle)


def _trainer(max_epochs: int = 2) -> lightning.pytorch.Trainer:
    return lightning.pytorch.Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )


def _predict(model: BaseDeep, loader: DataLoader) -> list[np.ndarray]:
    batches = _trainer().predict(model, loader)
    return [torch.cat(view).numpy() for view in zip(*batches)]


def _encoders(width: int = K) -> list[nn.Module]:
    return [nn.Linear(p, width) for p in P]


MODELS: dict[str, Callable[[], BaseDeep]] = {
    "DCCA": lambda: DCCA(K, _encoders()),
    "DCCAEY": lambda: DCCAEY(K, _encoders()),
    "DCCANOI": lambda: DCCANOI(K, _encoders()),
    "DCCASDL": lambda: DCCASDL(K, _encoders()),
    "DMCCA": lambda: DMCCA(K, _encoders()),
    "DGCCA": lambda: DGCCA(K, _encoders()),
    "DTCCA": lambda: DTCCA(K, _encoders()),
    "BarlowTwins": lambda: BarlowTwins(K, _encoders()),
    "VICReg": lambda: VICReg(K, _encoders()),
    "DCCAE": lambda: DCCAE(K, _encoders(), [nn.Linear(K, p) for p in P]),
    "SplitAE": lambda: SplitAE(K, _encoders(), [nn.Linear(2 * K, p) for p in P]),
}
TRAINABLE: dict[str, Callable[[], BaseDeep]] = {
    **MODELS,
    "DVCCA": lambda: DVCCA(K, nn.Linear(P[0], 2 * K), [nn.Linear(K, p) for p in P]),
}


@pytest.mark.parametrize("name", MODELS)
def test_predict_returns_canonical_variates(name: str) -> None:
    """Training ends with a linear CCA, so predictions are canonical variates."""
    model = MODELS[name]()
    views = _views()
    _trainer().fit(model, _loader(views, shuffle=True))
    scores = _predict(model, _loader(views))
    assert [s.shape for s in scores] == [(64, K), (64, K)]
    for s in scores:
        np.testing.assert_allclose(np.cov(s, rowvar=False), np.eye(K), atol=1e-3)
    corrs = pairwise_correlations(scores)[0, 1]
    assert corrs[0] >= corrs[1]


@pytest.mark.parametrize("name", TRAINABLE)
def test_validation_loss_is_the_training_loss(name: str) -> None:
    """Every loss term is logged for validation, reconstruction included."""
    model = TRAINABLE[name]()
    trainer = _trainer(max_epochs=1)
    trainer.fit(model, _loader(_views()), _loader(_views(seed=1)))
    terms = model.loss({"views": [torch.as_tensor(v) for v in _views()]})
    logged = {k for k in trainer.callback_metrics if k.startswith("val/")}
    assert logged == {f"val/{k}" for k in terms}
    assert float(trainer.callback_metrics["val/objective"]) != 0.0


def test_predict_before_fitting_raises() -> None:
    """Predicting needs the linear CCA fitted at the end of training."""
    with pytest.raises(RuntimeError, match="fit_cca"):
        _predict(DCCA(K, _encoders()), _loader(_views()))


def test_fit_cca_enables_prediction_without_training() -> None:
    """fit_cca fits the projection on any loader."""
    model = DCCA(K, _encoders())
    model.fit_cca(_loader(_views()))
    assert _predict(model, _loader(_views()))[0].shape == (64, K)


@pytest.mark.parametrize("name", ["DCCA", "DVCCA", "DCCAE"])
def test_checkpoint_round_trip(name: str, tmp_path: Path) -> None:
    """Hyperparameters and the fitted projection are restored from a checkpoint."""
    model = TRAINABLE[name]()
    trainer = _trainer()
    trainer.fit(model, _loader(_views()))
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    fresh = TRAINABLE[name]()
    modules: dict[str, object] = (
        {"encoder": fresh.encoders[0]}
        if name == "DVCCA"
        else {"encoders": list(fresh.encoders)}
    )
    if hasattr(fresh, "decoders"):
        modules["decoders"] = list(fresh.decoders)
    restored = type(model).load_from_checkpoint(path, **modules)
    for a, b in zip(
        _predict(model, _loader(_views())), _predict(restored, _loader(_views()))
    ):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_dcca_learns_shared_signal() -> None:
    """DCCA recovers the shared signal on held-out data."""
    torch.manual_seed(0)
    views = _views(n=400)
    model = DCCA(K, _encoders(), learning_rate=1e-2)
    _trainer(max_epochs=30).fit(model, _loader([v[:300] for v in views], shuffle=True))
    scores = _predict(model, _loader([v[300:] for v in views]))
    assert pairwise_correlations(scores)[0, 1].min() > 0.8


@pytest.mark.parametrize(
    "make",
    [
        lambda e: DCCA(K, e),
        lambda e: DCCAE(K, e, [nn.Linear(K, 4)] * 3),
        lambda e: DCCASDL(K, e),
        lambda e: BarlowTwins(K, e),
        lambda e: VICReg(K, e),
    ],
    ids=["DCCA", "DCCAE", "DCCASDL", "BarlowTwins", "VICReg"],
)
def test_two_view_models_reject_more_views(make: Callable) -> None:
    """Models defined for two views raise rather than ignore extra views."""
    with pytest.raises(ValueError, match="two views"):
        make([nn.Linear(4, K) for _ in range(3)])


@pytest.mark.parametrize(
    "make",
    [
        lambda e: DCCA(K, e, objective=MCCALoss()),
        lambda e: DMCCA(K, e),
        lambda e: DGCCA(K, e),
        lambda e: DTCCA(K, e),
        lambda e: DCCAEY(K, e),
    ],
    ids=["DCCA+MCCALoss", "DMCCA", "DGCCA", "DTCCA", "DCCAEY"],
)
def test_multiview_models_train_on_three_views(make: Callable) -> None:
    """Multiview losses train on three views and predict one array per view."""
    views = _views(n_views=3)
    model = make([nn.Linear(v.shape[1], K) for v in views])
    _trainer().fit(model, _loader(views))
    assert len(_predict(model, _loader(views))) == 3


def test_encoder_width_must_match_n_components() -> None:
    """An encoder whose output is not n_components wide raises."""
    model = DCCA(3, _encoders())
    with pytest.raises(ValueError, match="n_components"):
        model([torch.randn(4, p) for p in P])


def test_dvcca_encoder_width_must_be_twice_n_components() -> None:
    """The DVCCA encoder outputs a mean and a log-variance."""
    model = DVCCA(K, nn.Linear(P[0], K), [nn.Linear(K, p) for p in P])
    with pytest.raises(ValueError, match="2 \\* n_components"):
        model([torch.randn(4, p) for p in P])


def test_dvcca_predicts_the_first_views_posterior_mean() -> None:
    """DVCCA encodes the first view alone and predicts its posterior mean."""
    model = DVCCA(K, nn.Linear(P[0], 2 * K), [nn.Linear(K, p) for p in P])
    views = _views()
    _trainer().fit(model, _loader(views))
    (mean,) = _predict(model, _loader(views))
    expected = model.encoders[0](torch.as_tensor(views[0]))[:, :K]
    np.testing.assert_allclose(mean, expected.detach().numpy(), atol=1e-6)


def test_dcca_ey_uses_independent_views() -> None:
    """An independent batch changes the penalty estimate."""
    model = DCCAEY(K, _encoders())
    views = [torch.as_tensor(v) for v in _views()]
    other = [torch.as_tensor(v) for v in _views(seed=1)]
    plain = model.loss({"views": views})
    independent = model.loss({"views": views, "independent_views": other})
    assert torch.equal(plain["rewards"], independent["rewards"])
    assert not torch.equal(plain["penalties"], independent["penalties"])


def test_dcca_noi_whitens_with_running_covariance_in_eval() -> None:
    """In eval mode NOI's whitening uses the running covariance, unchanged."""
    model = DCCANOI(K, _encoders())
    z = torch.randn(32, K) * 3.0
    model.bws[0](z)
    running = model.bws[0].running_covar.clone()
    model.eval()
    out = model.bws[0](z)
    assert torch.equal(model.bws[0].running_covar, running)
    assert not torch.allclose(out, z)


def test_multiview_dataset_batches() -> None:
    """MultiviewDataset yields {"views": [...]} batches."""
    batch = next(iter(_loader(_views())))
    assert list(batch) == ["views"]
    assert [v.shape for v in batch["views"]] == [(32, 6), (32, 5)]


def test_objectives_are_scalars() -> None:
    """Each loss returns a scalar; CCALoss is non-positive and two-view only."""
    two = [torch.randn(16, 4) for _ in range(2)]
    three = [torch.randn(16, 4) for _ in range(3)]
    assert float(CCALoss(eps=1e-4)(two)) <= 0.0
    for loss in (MCCALoss(eps=1e-4), GCCALoss(eps=1e-4), TCCALoss(eps=1e-4)):
        assert loss(three).ndim == 0
    with pytest.raises(ValueError, match="exactly 2"):
        CCALoss()(three)
