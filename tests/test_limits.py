"""Each hyperparameter at its limits, against a model it must reduce to.

Without a penalty the penalised models are CCA; without sparsity the sparse
power iterations are sklearn's PLSCanonical; a large penalty zeroes every
weight, and sparsity only grows with the penalty.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSCanonical

from cca_zoo._base import BaseModel
from cca_zoo.linear import CCA, CCAEY, GCCA, PLS, HuberCCA
from cca_zoo.sparse import (
    ADMMCCA,
    IPLSCCA,
    PMDCCA,
    ElasticNetCCA,
    MultiTaskElasticNetCCA,
    OrthogonalMatchingPursuitCCA,
    ParkhomenkoCCA,
    SpanCCA,
)
from cca_zoo.stochastic import StochasticCCAEY


def _views() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 3)) * [3, 2, 1]
    return [
        z @ rng.standard_normal((3, p)) + rng.standard_normal((300, p)) for p in (6, 5)
    ]


def _subspace_cosines(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """Cosines of the principal angles between each view's two score subspaces."""
    return np.concatenate(
        [
            np.linalg.svd(
                np.linalg.qr(x - x.mean(axis=0))[0].T
                @ np.linalg.qr(y - y.mean(axis=0))[0],
                compute_uv=False,
            )
            for x, y in zip(a, b)
        ]
    )


@pytest.mark.parametrize(
    "model",
    [
        GCCA(2),
        HuberCCA(2, delta=1e6, random_state=0),
        ElasticNetCCA(2, alpha=0.0, random_state=0),
        MultiTaskElasticNetCCA(2, alpha=0.0, random_state=0),
        OrthogonalMatchingPursuitCCA(2, n_nonzero_coefs=[6, 5], random_state=0),
    ],
    ids=lambda m: type(m).__name__,
)
def test_without_a_penalty_is_cca(model: BaseModel) -> None:
    """With no penalty, trimming or budget, the scores span CCA's."""
    views = _views()
    np.testing.assert_allclose(
        _subspace_cosines(
            model.fit(views).transform(views), CCA(2).fit(views).transform(views)
        ),
        1.0,
        atol=1e-3,
    )


@pytest.mark.parametrize("cls", [CCAEY, StochasticCCAEY])
@pytest.mark.parametrize("units", [1.0, 100.0])
def test_full_shrinkage_is_pls(cls: type, units: float) -> None:
    """At shrinkage=1 the EY models find PLS's subspace, in any units."""
    views = _views()
    views = [views[0] * units, views[1]]
    model = cls(2, shrinkage=1.0, max_iter=5000, random_state=0).fit(views)
    np.testing.assert_allclose(
        _subspace_cosines(model.transform(views), PLS(2).fit(views).transform(views)),
        1.0,
        atol=0.05,
    )


@pytest.mark.parametrize(
    ("batch_size", "atol"), [(None, 1e-3), (128, 0.01), (32, 0.01)]
)
def test_stochastic_reaches_cca_at_any_batch_size(
    batch_size: int | None, atol: float
) -> None:
    """At its defaults, SGD settles on CCA's subspace rather than near it.

    Full-batch it converges; mini-batches settle within their noise. The
    views are ill-conditioned, and their canonical correlations (0.98, 0.91,
    0.77) leave the two-dimensional subspace identifiable.
    """
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1000, 3)) * [2.0, 1.0, 0.5]
    views = [
        z @ rng.standard_normal((3, p)) + rng.standard_normal((1000, p))
        for p in (20, 15)
    ]
    model = StochasticCCAEY(2, batch_size=batch_size, random_state=0).fit(views)
    np.testing.assert_allclose(
        _subspace_cosines(model.transform(views), CCA(2).fit(views).transform(views)),
        1.0,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("model", "scale"),
    [
        (PMDCCA(3, l1_bound=1.0, random_state=0), False),
        (SpanCCA(3, random_state=0), False),
        (ParkhomenkoCCA(3, alpha=0.0, random_state=0), True),
    ],
    ids=["PMDCCA", "SpanCCA", "ParkhomenkoCCA"],
)
def test_without_sparsity_is_pls_canonical(model: BaseModel, scale: bool) -> None:
    """Unthresholded, the power iterations deflate as PLSCanonical does."""
    views = _views()
    ours = model.fit(views).transform(views)
    theirs = PLSCanonical(3, scale=scale).fit(*views).transform(*views)
    for x, y in zip(ours, theirs):
        correlations = [abs(np.corrcoef(x[:, d], y[:, d])[0, 1]) for d in range(3)]
        np.testing.assert_allclose(correlations, 1.0, atol=1e-5)


@pytest.mark.parametrize(
    ("cls", "name", "strengthening"),
    [
        (ElasticNetCCA, "alpha", [0.001, 0.05, 0.2, 1.0, 100.0]),
        (MultiTaskElasticNetCCA, "alpha", [0.001, 0.05, 0.2, 1.0, 100.0]),
        (IPLSCCA, "alpha", [0.001, 0.05, 0.2, 1.0, 100.0]),
        (ADMMCCA, "alpha", [0.001, 0.05, 0.2, 1.0, 100.0]),
        (ParkhomenkoCCA, "alpha", [0.0, 0.3, 0.6, 0.9, 1.0]),
        (PMDCCA, "l1_bound", [1.0, 0.6, 0.4, 0.2, 0.1]),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_sparsity_grows_with_the_penalty_until_every_weight_is_zero(
    cls: type, name: str, strengthening: list[float]
) -> None:
    """Nonzero weights never increase along the path, and end at none."""
    views = _views()
    nonzero = [
        sum(
            int(np.sum(np.abs(w) > 1e-10))
            for w in cls(**{name: value}, random_state=0).fit(views).weights_
        )
        for value in strengthening
    ]
    assert nonzero == sorted(nonzero, reverse=True)
    assert nonzero[-1] == 0
