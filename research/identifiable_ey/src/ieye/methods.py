"""Estimators. Each maps two views to weights ``[W_1, W_2]`` of shape (p, k) on raw features."""

from __future__ import annotations

import numpy as np
import torch
from cca_zoo.linear import CCA, CCAEY
from multiviewica import multiviewica
from sklearn.decomposition import FastICA

from ieye.loss import (
    SecondMoments,
    cf_distance,
    Variate,
    gauss_hermite,
    legendre,
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


def _quadrature(denoise: bool, n_nodes: int, bandwidth: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal-weighted nodes, or a bounded range ``[0, bandwidth]`` when deconvolving."""
    return legendre(n_nodes, bandwidth) if denoise else gauss_hermite(n_nodes, bandwidth)


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
    denoise: bool = False,
) -> list[np.ndarray]:
    """Full-batch joint minimisation of ``L_EY - mu * sum NG`` from rotated CCA starts.

    The best of ``restarts`` random common rotations of the CCA solution, by
    final objective (unsupervised), is returned.
    """
    white = Whitened(views)
    nodes, weights = _quadrature(denoise, n_nodes, bandwidth)
    batch = [torch.tensor(x) for x in white.x]
    rng = np.random.default_rng(seed)
    start = _cca_start(white, k)
    best_value, best = np.inf, None
    for _ in range(restarts):
        q = _random_rotation(k, rng)
        w = [torch.tensor(s @ q).requires_grad_() for s in start]

        def objective(w=w) -> torch.Tensor:
            return identifiable_ey_loss(
                batch, w, white.moments, mu=mu, variate=variate, nodes=nodes, weights=weights, denoise=denoise
            )["objective"]

        value = _minimise(objective, w, max_iter)
        if value < best_value:
            best_value, best = value, white.raw_weights(w)
    return best


def rotation_only(
    views: list[np.ndarray], k: int, seed: int, *, variate: Variate = "average",
    restarts: int = 8, n_nodes: int = 17, bandwidth: float = 1.0, max_iter: int = 500,
    denoise: bool = False,
) -> list[np.ndarray]:
    """CCA, then the orthogonal rotation maximising the CF non-Gaussianity of the shared variate.

    A rotation of the whitened-embedding preserves the EY value, so only NG
    selects it; its standardisation is exact because the CCA variates are
    scaled by their population standard deviations.
    """
    white = Whitened(views)
    nodes, weights = _quadrature(denoise, n_nodes, bandwidth)
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
            return -non_gaussianity(batch, rotated(), white.moments, variate, nodes, weights, denoise)

        value = _minimise(objective, [a], max_iter)
        if value < best_value:
            best_value, best = value, white.raw_weights(rotated())
    return best


class Overselected:
    """The shared average of the top ``m`` CCA components, whitened to identity.

    ``u = mean_v (x_v - mean) W_v T`` has identity covariance on the data, so
    any orthonormal ``(m, k)`` matrix ``Q`` gives ``k`` unit-variance,
    uncorrelated components ``u Q`` whose standardisation is exact.
    """

    def __init__(self, views: list[np.ndarray], m: int, seed: int) -> None:
        self.weights = cca(views, m, seed)
        shared = np.mean([(x - x.mean(0)) @ w for x, w in zip(views, self.weights)], axis=0)
        vals, vecs = np.linalg.eigh(np.cov(shared.T))
        self.whiten = vecs / np.sqrt(vals)
        self.u = shared @ self.whiten

    def view_weights(self, q: np.ndarray) -> list[np.ndarray]:
        """Raw-feature weights of the components ``u Q``."""
        return [w @ self.whiten @ q for w in self.weights]


def overselect_fastica(
    views: list[np.ndarray], k: int, seed: int, *, m: int
) -> list[np.ndarray]:
    """Top-``m`` CCA, FastICA to ``m`` components, keep the ``k`` most non-Gaussian.

    Non-Gaussianity is the squared gap of ``E log cosh`` to its Gaussian value.
    """
    over = Overselected(views, m, seed)
    ica = FastICA(n_components=m, whiten=False, random_state=seed, max_iter=1000)
    sources = ica.fit_transform(over.u)
    sources /= sources.std(0)
    gaussian = np.log(np.cosh(np.random.default_rng(0).standard_normal(1_000_000))).mean()
    score = (np.log(np.cosh(sources)).mean(0) - gaussian) ** 2
    return over.view_weights(ica.components_.T[:, np.argsort(score)[::-1][:k]])


def overselect_pursuit(
    views: list[np.ndarray], k: int, seed: int, *, m: int, restarts: int = 8,
    n_nodes: int = 17, bandwidth: float = 1.0, max_iter: int = 300,
) -> list[np.ndarray]:
    """Top-``m`` CCA, then the orthonormal ``(m, k)`` frame maximising CF non-Gaussianity.

    The frame is the Q factor of a free ``(m, k)`` matrix. The restart with the
    highest objective is returned; one restart starts at the top-``k`` CCA frame.
    """
    over = Overselected(views, m, seed)
    nodes, weights = gauss_hermite(n_nodes, bandwidth)
    u = torch.tensor(over.u)
    rng = np.random.default_rng(seed)
    starts = [np.eye(m, k)] + [rng.standard_normal((m, k)) for _ in range(restarts - 1)]
    best_value, best = np.inf, None
    for start in starts:
        free = torch.tensor(start).requires_grad_()

        def frame(free=free) -> torch.Tensor:
            return torch.linalg.qr(free)[0]

        def objective(frame=frame) -> torch.Tensor:
            return -cf_distance(u @ frame(), nodes, weights).sum()

        value = _minimise(objective, [free], max_iter)
        if value < best_value:
            best_value, best = value, frame().detach().numpy()
    return over.view_weights(best)
