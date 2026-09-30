"""The deep models, as Lightning modules. Slow; need torch and lightning."""

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
    NRDCCA,
    BarlowTwins,
    BaseDeep,
    DVCCAPrivate,
    LeJEPA,
    MultiviewDataset,
    SplitAE,
    VICReg,
)
from cca_zoo.deep._dcca_ey import _cca_cv
from cca_zoo.deep._lejepa import _sigreg
from cca_zoo.deep._nrdcca import _mean_canonical_correlation
from cca_zoo.deep.objectives import CCALoss, GCCALoss, MCCALoss, TCCALoss
from cca_zoo.linear import CCA, GCCA, MCCA, TCCA
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
    "LeJEPA": lambda: LeJEPA(K, _encoders(), n_slices=16),
    "NRDCCA": lambda: NRDCCA(K, _encoders()),
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
def test_predict_returns_the_encodings(name: str) -> None:
    """trainer.predict returns what calling the model does, one array per view."""
    model = MODELS[name]()
    views = _views()
    _trainer().fit(model, _loader(views, shuffle=True))
    model.eval()
    expected = model([torch.as_tensor(v) for v in views])
    for scores, z in zip(_predict(model, _loader(views)), expected):
        np.testing.assert_allclose(scores, z.detach().numpy(), atol=1e-6)


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
    trainer.test(model, _loader(_views(seed=1)), verbose=False)
    assert "test/objective" in trainer.callback_metrics


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
    train = _loader([v[:300] for v in views], shuffle=True)
    _trainer(max_epochs=30).fit(model, train)
    cca = CCA(n_components=K).fit(_predict(model, _loader([v[:300] for v in views])))
    scores = cca.transform(_predict(model, _loader([v[300:] for v in views])))
    assert pairwise_correlations(scores)[0, 1].min() > 0.8


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
        "LeJEPA": LeJEPA(K, enc(), n_slices=16),
        "NRDCCA": NRDCCA(K, enc()),
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


def test_dvcca_predicts_the_first_views_posterior_mean() -> None:
    """DVCCA encodes the first view alone and predicts its posterior mean."""
    model = DVCCA(K, nn.Linear(P[0], 2 * K), [nn.Linear(K, p) for p in P])
    views = _views()
    _trainer().fit(model, _loader(views))
    (mean,) = _predict(model, _loader(views))
    expected = model.encoders[0](torch.as_tensor(views[0]))[:, :K]
    np.testing.assert_allclose(mean, expected.detach().numpy(), atol=1e-6)


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


def test_encoder_widths_are_checked() -> None:
    """Constructor arguments and encoder widths are checked."""
    for model, argument in [(DCCANOI, "rho"), (DPCCA, "rho"), (DCCAE, "lam")]:
        kwargs = {argument: 2.0}
        if model is DCCAE:
            kwargs["decoders"] = [nn.Linear(K, p) for p in P]
        with pytest.raises(ValueError, match=argument):
            model(K, _encoders(), **kwargs)
    with pytest.raises(ValueError, match="two views"):
        DCCA(K, [nn.Linear(4, K) for _ in range(3)])
    with pytest.raises(ValueError, match="n_components"):
        DCCA(3, _encoders())([torch.randn(4, p) for p in P])
    with pytest.raises(ValueError, match="2 \\* n_components"):
        DVCCA(K, nn.Linear(P[0], K), [nn.Linear(K, p) for p in P])(
            [torch.randn(4, p) for p in P]
        )
    private = DVCCAPrivate(
        K,
        nn.Linear(P[0], 2 * K),
        [nn.Linear(p, 1) for p in P],
        [nn.Linear(K + 1, p) for p in P],
        n_private=1,
    )
    with pytest.raises(ValueError, match="2 \\* n_private"):
        private.loss({"views": [torch.randn(4, p) for p in P]})


def _linear_views(n_views: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((500, 3)) * [3, 2, 1]
    return [
        z @ rng.standard_normal((3, p)) + rng.standard_normal((500, p))
        for p in (6, 5, 4)[:n_views]
    ]


def _ey_loss(representations: list[torch.Tensor]) -> torch.Tensor:
    """DCCAEY's objective."""
    c, v = _cca_cv(representations)
    return -torch.trace(2.0 * c) + torch.trace(v @ v)


def _minimised_over_linear_encoders(
    loss: Callable[[list[torch.Tensor]], torch.Tensor],
    views: list[np.ndarray],
    k: int,
) -> list[np.ndarray]:
    """The encodings of the linear maps minimising ``loss``, by full-batch L-BFGS."""
    torch.manual_seed(0)
    xs = [torch.tensor(v - v.mean(axis=0)) for v in views]
    weights = [
        torch.randn(x.shape[1], k, dtype=x.dtype, requires_grad=True) for x in xs
    ]
    optimiser = torch.optim.LBFGS(
        weights,
        max_iter=5000,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        value = loss([x @ w for x, w in zip(xs, weights)])
        value.backward()
        return value

    for _ in range(5):
        optimiser.step(closure)
    return [(x @ w).detach().numpy() for x, w in zip(xs, weights)]


def _same_subspace(
    a: list[np.ndarray], b: list[np.ndarray], atol: float = 1e-4
) -> None:
    for x, y in zip(a, b):
        qx = np.linalg.qr(x - x.mean(axis=0))[0]
        qy = np.linalg.qr(y - y.mean(axis=0))[0]
        np.testing.assert_allclose(
            np.linalg.svd(qx.T @ qy, compute_uv=False), 1.0, atol=atol
        )


@pytest.mark.parametrize(
    ("loss", "linear", "n_views"),
    [
        (CCALoss(eps=1e-10), CCA(2), 2),
        (MCCALoss(eps=1e-10), CCA(2), 2),
        (_ey_loss, CCA(2), 2),
        (_ey_loss, MCCA(2), 3),
        (GCCALoss(eps=1e-10), GCCA(2), 3),
    ],
    ids=["CCALoss", "MCCALoss", "EY", "EY-3-views", "GCCALoss"],
)
def test_linear_encoders_minimising_a_loss_are_its_linear_model(
    loss: Callable[[list[torch.Tensor]], torch.Tensor],
    linear: object,
    n_views: int,
) -> None:
    """Each deep objective, over linear encoders, is minimised by its linear model."""
    views = _linear_views(n_views)
    _same_subspace(
        _minimised_over_linear_encoders(loss, views, 2),
        linear.fit(views).transform(views),  # type: ignore[attr-defined]
    )


def test_tcca_attains_the_minimum_of_its_loss() -> None:
    """No linear encoding scores below TCCA's on TCCALoss.

    The tensor loss has local minima, so a single descent may stop above it.
    """
    views = _linear_views(3)
    loss = TCCALoss(eps=1e-10)
    tcca = [
        torch.tensor(z) for z in TCCA(1, random_state=0).fit(views).transform(views)
    ]
    descended = [
        torch.tensor(z) for z in _minimised_over_linear_encoders(loss, views, 1)
    ]
    assert float(loss(tcca)) <= float(loss(descended)) + 1e-8


@pytest.mark.parametrize("batch_size", [16, 256])
def test_gcca_loss_is_minus_k_per_view_when_the_encodings_agree(
    batch_size: int,
) -> None:
    """Every view's projection is the same, so the top k eigenvalues are M each."""
    z = torch.randn(batch_size, 2, dtype=torch.float64)
    loss = GCCALoss(eps=1e-10)(
        [
            z,
            2.0 * z + 1.0,
            z @ torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        ]
    )
    assert float(loss) == pytest.approx(-2 * 3)


@pytest.mark.parametrize("cls", [DCCANOI, DCCASDL])
def test_trained_linear_encoders_reach_cca(cls: type[BaseDeep]) -> None:
    """With linear encoders, full-batch training converges to CCA's subspace."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((500, 2)) * [1.0, 0.6]
    views = [
        (z @ rng.standard_normal((2, p)) + 0.5 * rng.standard_normal((500, p))).astype(
            np.float32
        )
        for p in P
    ]
    torch.manual_seed(0)
    model = cls(K, _encoders(), learning_rate=1e-2)
    full_batch = DataLoader(MultiviewDataset(views), batch_size=len(views[0]))
    _trainer(max_epochs=300).fit(model, full_batch)
    for ours, cca in zip(
        _predict(model, full_batch), CCA(K).fit(views).transform(views)
    ):
        qx = np.linalg.qr(ours - ours.mean(axis=0))[0]
        qy = np.linalg.qr(cca - cca.mean(axis=0))[0]
        np.testing.assert_allclose(
            np.linalg.svd(qx.T @ qy, compute_uv=False), 1.0, atol=5e-3
        )


def test_cca_loss_is_minus_the_squared_canonical_correlations() -> None:
    """At CCA's scores, whatever each view's invertible mixing of them."""
    views = _linear_views(2)
    model = CCA(2).fit(views)
    scores = [torch.tensor(z) for z in model.transform(views)]
    squared = np.sum(pairwise_correlations(model.transform(views))[0, 1] ** 2)
    mixing = torch.tensor(np.random.default_rng(1).standard_normal((2, 2)))
    for pair in (scores, [scores[0] @ mixing, scores[1]]):
        assert float(CCALoss(eps=1e-10)(pair)) == pytest.approx(-squared, abs=1e-8)
    with pytest.raises(ValueError, match="exactly 2"):
        CCALoss()([scores[0]] * 3)


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
@pytest.mark.parametrize("seed", [0, 4])
def test_dpcca_finds_the_signal_the_partials_do_not_explain(
    encode_partials: bool, seed: int
) -> None:
    """Conditioned on the partials, DPCCA recovers the signal they do not explain."""
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


def test_sigreg_is_small_for_a_standard_normal_and_large_for_a_collapse() -> None:
    """SIGReg tests each random projection against a standard normal."""
    torch.manual_seed(0)
    gaussian = _sigreg(torch.randn(2048, 8), 0, 64, 17, 5.0)
    collapsed = _sigreg(torch.zeros(2048, 8), 0, 64, 17, 5.0)
    assert float(gaussian) < 2.0
    assert float(collapsed) > 100 * float(gaussian)


def test_sigreg_directions_follow_the_seed() -> None:
    """Directions repeat for a seed and are resampled across seeds."""
    z = torch.randn(64, 4) * torch.tensor([3.0, 1.0, 0.2, 1.0])
    assert torch.equal(_sigreg(z, 1, 8, 17, 5.0), _sigreg(z, 1, 8, 17, 5.0))
    assert not torch.equal(_sigreg(z, 1, 8, 17, 5.0), _sigreg(z, 2, 8, 17, 5.0))


def test_lejepa_views_that_agree_leave_only_sigreg() -> None:
    """Identical encodings have no predictive loss, and lam weighs SIGReg."""
    model = LeJEPA(K, [nn.Identity(), nn.Identity()], lam=0.3)
    z = torch.randn(16, K)
    terms = model.loss({"views": [z, z.clone()]})
    assert float(terms["sim_loss"]) == 0.0
    torch.testing.assert_close(terms["objective"], 0.3 * terms["sigreg"])


def test_noise_correlation_is_invariant_to_an_invertible_linear_map() -> None:
    """NR-DCCA's Theorem 1: Corr(X, A) = Corr(XW, AW) for linear CCA."""
    torch.manual_seed(0)
    x = torch.randn(200, 6, dtype=torch.float64) @ torch.randn(
        6, 6, dtype=torch.float64
    )
    a = torch.randn(200, 6, dtype=torch.float64)
    w = torch.randn(6, 6, dtype=torch.float64)
    torch.testing.assert_close(
        _mean_canonical_correlation(x @ w, a @ w, 1e-12),
        _mean_canonical_correlation(x, a, 1e-12),
    )


def test_nrdcca_without_regularisation_is_dmcca() -> None:
    """At alpha=0 the objective is DMCCA's."""
    views = [torch.randn(32, p) for p in P]
    encoders = _encoders()
    nr = NRDCCA(K, encoders, alpha=0.0).loss({"views": views})
    dmcca = DMCCA(K, encoders).loss({"views": views})
    torch.testing.assert_close(nr["objective"], dmcca["objective"])


def test_linear_lejepa_is_cca() -> None:
    """With little SIGReg, linear encoders minimising LeJEPA's loss span CCA's.

    SIGReg then acts as a whitening constraint, under which the distance to
    the views' centre is CCA's objective.
    """
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 2)) * [1.0, 0.6]
    views = [
        z @ rng.standard_normal((2, p)) + 0.5 * rng.standard_normal((300, p)) for p in P
    ]
    xs = [torch.tensor(v - v.mean(axis=0)) for v in views]
    model = LeJEPA(K, [nn.Identity(), nn.Identity()], lam=0.05, n_slices=16)
    torch.manual_seed(0)
    weights = [
        torch.randn(x.shape[1], K, dtype=x.dtype, requires_grad=True) for x in xs
    ]
    optimiser = torch.optim.Adam(weights, lr=3e-2)
    for _ in range(2000):
        optimiser.zero_grad()
        model.loss({"views": [x @ w for x, w in zip(xs, weights)]})[
            "objective"
        ].backward()
        optimiser.step()
    _same_subspace(
        [(x @ w).detach().numpy() for x, w in zip(xs, weights)],
        CCA(K).fit(views).transform(views),
        atol=5e-3,
    )
