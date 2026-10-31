"""Estimators. Each maps two views to weights ``[W_1, W_2]`` of shape (p, k) on raw features."""

from __future__ import annotations

import numpy as np
import torch
from cca_zoo.linear import CCA, CCAEY
from multiviewica import multiviewica
from sklearn.decomposition import FastICA

from ieye.loss import (
    SecondMoments,
    Variate,
    gauss_hermite,
    identifiable_ey_loss,
    non_gaussianity,
)


def cca(views: list[np.ndarray], k: int, seed: int) -> list[np.ndarray]:
    """Exact CCA."""
    return list(CCA(n_components=k).fit(views).weights_)


def ey(views: list[np.ndarray], k: int, seed: int) -> list[np.ndarray]:
    """Full-batch EY."""
    return list(CCAEY(n_components=k, random_state=seed).fit(views).weights_)


def cca_fastica(
    views: list[np.ndarray], k: int, seed: int, fun: str = "logcosh"
) -> list[np.ndarray]:
    """CCA, then FastICA on the shared average ``(z_1 + z_2) / 2``."""
    weights = cca(views, k, seed)
    shared = np.mean([x @ w for x, w in zip(views, weights)], axis=0)
    ica = FastICA(n_components=k, fun=fun, whiten="unit-variance", random_state=seed, max_iter=1000)
    ica.fit(shared)
    return [w @ ica.components_.T for w in weights]


def multiview_ica(views: list[np.ndarray], k: int, seed: int) -> list[np.ndarray]:
    """CCA to ``k`` dimensions, then MultiViewICA (Richard et al. 2020) on the projections.

    MultiViewICA assumes square unmixing. Its PyPI release's SRM reduction is
    broken against the installed fastsrm, and PCA reduction keeps the
    highest-variance directions, not the shared ones, so CCA does the
    reduction. The projections are centred by CCA, which centres its inputs.
    """
    weights = cca(views, k, seed)
    projected = [(x - x.mean(0)) @ w for x, w in zip(views, weights)]
    projections, unmixings, _ = multiviewica(
        np.stack([z.T for z in projected]),
        n_components=k, dimension_reduction="pca", random_state=seed, max_iter=2000,
    )
    return [w @ (u @ p).T for w, u, p in zip(weights, unmixings, projections)]


class Whitened:
    """Views centred and whitened on the full data, with the maps back to raw features."""

    def __init__(self, views: list[np.ndarray]) -> None:
        self.means = [x.mean(0) for x in views]
        self.maps = []
        for x, m in zip(views, self.means):
            vals, vecs = np.linalg.eigh(np.cov(x.T))
            self.maps.append(vecs / np.sqrt(vals))
        self.x = [(x - m) @ t for x, m, t in zip(views, self.means, self.maps)]
        self.moments = SecondMoments.of(self.x)

    def raw_weights(self, w: list[torch.Tensor]) -> list[np.ndarray]:
        """Weights on whitened features mapped to raw features."""
        return [t @ e.detach().numpy() for t, e in zip(self.maps, w)]


def _cca_start(white: Whitened, k: int) -> list[np.ndarray]:
    """EY-scaled canonical weights in whitened coordinates (variance ``1 + rho``)."""
    n = white.x[0].shape[0]
    u, s, vt = np.linalg.svd(white.x[0].T @ white.x[1] / (n - 1))
    scale = np.sqrt(1 + s[:k])
    return [u[:, :k] * scale, vt[:k].T * scale]


def _random_rotation(k: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((k, k)))
    return q * np.sign(np.diag(r))


def _minimise(
    objective, params: list[torch.Tensor], max_iter: int
) -> float:
    """Full-batch L-BFGS; returns the final objective."""
    opt = torch.optim.LBFGS(
        params, lr=1.0, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=20, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(objective())


def joint(
    views: list[np.ndarray], k: int, seed: int, *, mu: float, variate: Variate = "average",
    restarts: int = 8, n_nodes: int = 17, bandwidth: float = 1.0, max_iter: int = 500,
) -> list[np.ndarray]:
    """Full-batch joint minimisation of ``L_EY - mu * sum NG`` from rotated CCA starts.

    The best of ``restarts`` random common rotations of the CCA solution, by
    final objective (unsupervised), is returned.
    """
    white = Whitened(views)
    nodes, weights = gauss_hermite(n_nodes, bandwidth)
    batch = [torch.tensor(x) for x in white.x]
    rng = np.random.default_rng(seed)
    start = _cca_start(white, k)
    best_value, best = np.inf, None
    for _ in range(restarts):
        q = _random_rotation(k, rng)
        w = [torch.tensor(s @ q).requires_grad_() for s in start]

        def objective(w=w) -> torch.Tensor:
            return identifiable_ey_loss(
                batch, w, white.moments, mu=mu, variate=variate, nodes=nodes, weights=weights
            )["objective"]

        value = _minimise(objective, w, max_iter)
        if value < best_value:
            best_value, best = value, white.raw_weights(w)
    return best


def rotation_only(
    views: list[np.ndarray], k: int, seed: int, *, variate: Variate = "average",
    restarts: int = 8, n_nodes: int = 17, bandwidth: float = 1.0, max_iter: int = 500,
) -> list[np.ndarray]:
    """CCA, then the orthogonal rotation maximising the CF non-Gaussianity of the shared variate.

    A rotation of the whitened-embedding preserves the EY value, so only NG
    selects it; its standardisation is exact because the CCA variates are
    scaled by their population standard deviations.
    """
    white = Whitened(views)
    nodes, weights = gauss_hermite(n_nodes, bandwidth)
    batch = [torch.tensor(x) for x in white.x]
    rng = np.random.default_rng(seed)
    start = [torch.tensor(s) for s in _cca_start(white, k)]
    best_value, best = np.inf, None
    for _ in range(restarts):
        a = torch.tensor(rng.standard_normal((k, k)) * 0.5).requires_grad_()

        def rotated(a=a) -> list[torch.Tensor]:
            q = torch.linalg.matrix_exp(a - a.T)
            return [s @ q for s in start]

        def objective(rotated=rotated) -> torch.Tensor:
            return -non_gaussianity(batch, rotated(), white.moments, variate, nodes, weights)

        value = _minimise(objective, [a], max_iter)
        if value < best_value:
            best_value, best = value, white.raw_weights(rotated())
    return best
