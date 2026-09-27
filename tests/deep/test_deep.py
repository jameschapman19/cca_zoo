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
    DPCCA,
    DTCCA,
    DVCCA,
    BarlowTwins,
    BaseDeep,
    DVCCAPrivate,
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
    "DVCCAPrivate": lambda: DVCCAPrivate(
        K,
        nn.Linear(P[0], 2 * K),
        [nn.Linear(p, 2) for p in P],
        [nn.Linear(K + 1, p) for p in P],
        n_private=1,
    ),
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


@pytest.mark.parametrize("name", ["DCCA", "DVCCA", "DVCCAPrivate", "DCCAE"])
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
        if isinstance(fresh, DVCCA)
        else {"encoders": list(fresh.encoders)}
    )
    for attr in ("decoders", "private_encoders"):
        if hasattr(fresh, attr):
            modules[attr] = list(getattr(fresh, attr))
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


def test_dcca_rejects_more_views() -> None:
    """DCCA is two-view, like linear CCA; DMCCA, DGCCA and DTCCA take more."""
    with pytest.raises(ValueError, match="two views"):
        DCCA(K, [nn.Linear(4, K) for _ in range(3)])


def _three_view_models(widths: list[int]) -> dict[str, BaseDeep]:
    """Every deep model built for three views of the given widths."""

    def enc() -> list[nn.Module]:
        return [nn.Linear(p, K) for p in widths]

    def dec(width: int) -> list[nn.Module]:
        return [nn.Linear(width, p) for p in widths]

    return {
        "DMCCA": DMCCA(K, enc()),
        "DGCCA": DGCCA(K, enc()),
        "DTCCA": DTCCA(K, enc()),
        "DCCAEY": DCCAEY(K, enc()),
        "DCCANOI": DCCANOI(K, enc()),
        "DCCASDL": DCCASDL(K, enc()),
        "BarlowTwins": BarlowTwins(K, enc()),
        "VICReg": VICReg(K, enc()),
        "DCCAE": DCCAE(K, enc(), dec(K)),
        "SplitAE": SplitAE(K, enc(), dec(3 * K)),
        "DVCCA": DVCCA(K, nn.Linear(widths[0], 2 * K), dec(K)),
        "DVCCAPrivate": DVCCAPrivate(
            K,
            nn.Linear(widths[0], 2 * K),
            [nn.Linear(p, 2) for p in widths],
            dec(K + 1),
            n_private=1,
        ),
    }


THREE_VIEW_NAMES = list(_three_view_models([6, 5, 4]))


@pytest.mark.parametrize("name", THREE_VIEW_NAMES)
def test_models_train_on_three_views(name: str) -> None:
    """Every model but DCCA takes any number of views."""
    views = _views(n_views=3)
    model = _three_view_models([v.shape[1] for v in views])[name]
    _trainer().fit(model, _loader(views))
    expected = 1 if name.startswith("DVCCA") else 3
    assert len(_predict(model, _loader(views))) == expected


@pytest.mark.parametrize("name", ["DCCASDL", "BarlowTwins", "VICReg"])
def test_pairwise_losses_use_every_view(name: str) -> None:
    """The third view changes the loss rather than being ignored."""
    views = [torch.as_tensor(v) for v in _views(n_views=3)]
    model = _three_view_models([v.shape[1] for v in views])[name]
    model.eval()
    changed = [views[0], views[1], views[2] * 3.0 + 1.0]
    assert not torch.equal(
        model.loss({"views": views})["objective"],
        model.loss({"views": changed})["objective"],
    )


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


def test_dvcca_private_encoder_width_must_be_twice_n_private() -> None:
    """Each private encoder outputs a mean and a log-variance."""
    model = DVCCAPrivate(
        K,
        nn.Linear(P[0], 2 * K),
        [nn.Linear(p, 1) for p in P],
        [nn.Linear(K + 1, p) for p in P],
        n_private=1,
    )
    with pytest.raises(ValueError, match="2 \\* n_private"):
        model.loss({"views": [torch.randn(4, p) for p in P]})


def test_dvcca_private_means_come_from_each_view() -> None:
    """Each private posterior mean is the first half of its encoder's output."""
    model = TRAINABLE["DVCCAPrivate"]()
    assert isinstance(model, DVCCAPrivate)
    views = [torch.as_tensor(v) for v in _views()]
    for mean, enc, v in zip(model.private_means(views), model.private_encoders, views):
        torch.testing.assert_close(mean, enc(v)[:, :1])


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


def _confounded_views(n: int = 1500) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Views sharing a strong signal carried by the partials and a weak one not."""
    rng = np.random.default_rng(0)
    s, p = rng.standard_normal(n), rng.standard_normal(n)
    views = []
    for width in P:
        x = 0.3 * rng.standard_normal((n, width))
        x[:, :2] += np.outer(2.0 * s, rng.standard_normal(2))
        x[:, 2:4] += np.outer(p, rng.standard_normal(2))
        views.append(x.astype(np.float32))
    partials = np.column_stack([s, rng.standard_normal(n)]).astype(np.float32)
    return views, partials, p


@pytest.mark.parametrize("encode_partials", [False, True], ids=["raw", "net"])
@pytest.mark.parametrize("seed", [0, 4, 5])
def test_dpcca_finds_the_signal_the_partials_do_not_explain(
    encode_partials: bool, seed: int
) -> None:
    """Conditioned on the partials, DPCCA recovers the unconfounded signal.

    Seeds 4 and 5 are those where a partial encoder trained on the correlation
    loss collapsed.
    """
    torch.manual_seed(seed)
    partial_encoder = nn.Linear(2, 2) if encode_partials else None
    views, partials, p = _confounded_views()
    model = DPCCA(
        1,
        [nn.Sequential(nn.Linear(w, 16), nn.Tanh(), nn.Linear(16, 1)) for w in P],
        partial_encoder=partial_encoder,
        learning_rate=1e-2,
    )
    train = DataLoader(
        MultiviewDataset(views, partials=partials), batch_size=256, shuffle=True
    )
    _trainer(max_epochs=30).fit(model, train)
    (z1, _) = _predict(model, _loader(views))  # no partials needed to predict
    assert abs(np.corrcoef(z1[:, 0], p)[0, 1]) > 0.8


def test_dpcca_needs_partials_to_train() -> None:
    """A training batch without partials raises."""
    model = DPCCA(K, _encoders())
    with pytest.raises(ValueError, match="partials"):
        model.loss({"views": [torch.randn(8, p) for p in P]})
