r"""The identifiable-EY loss.

Two views, whitened and centred once on the full data. For weights
$W_v \in \mathbb{R}^{p \times k}$ and embeddings $Z_v = X_v W_v$:

$$
\mathcal{L} = \mathcal{L}_{EY} - \mu \sum_{i=1}^k \mathrm{NG}(\tilde z_i),
\qquad
\mathrm{NG}(z) = \int |\varphi_z(t) - e^{-t^2/2}|^2 \, \omega(dt),
$$

with $\omega$ the standard normal law (Gauss-Hermite nodes) and
$\tilde z_i = z_i / \sigma_i(W)$ the $i$-th shared variate standardised by
its *population* standard deviation, computed from fixed second-moment
blocks as a differentiable function of $W$. No minibatch statistic enters
the standardisation, so it adds no bias. Per-batch estimators:

* ``|phi|^2`` by the off-diagonal U-statistic $(n |\hat\varphi|^2 - 1)/(n-1)$,
* ``Re phi`` by the sample mean of ``cos(t z)``,
* the EY penalty ``tr(V V')`` with ``V'`` from an independent batch.

Each is unbiased for fixed $W$.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

Variate = Literal["average", "views"]


def gauss_hermite(n_nodes: int, bandwidth: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes ``t`` and weights ``w`` with ``sum_m w_m f(t_m) ~ E f(bandwidth * N(0,1))``."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    return (
        torch.tensor(bandwidth * x, dtype=torch.float64),
        torch.tensor(w / np.sqrt(2 * np.pi), dtype=torch.float64),
    )


def cf_distance(
    z: torch.Tensor, nodes: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Unbiased estimate of ``NG`` for each column of unit-variance ``z``, shape (k,).

    The constant ``E_w exp(-t^2)`` is included so the value is the squared
    distance itself, which is non-negative in expectation.
    """
    n = z.shape[0]
    angle = z[:, :, None] * nodes  # (n, k, T)
    cos_sum, sin_sum = torch.cos(angle).sum(0), torch.sin(angle).sum(0)
    abs_phi_sq = ((cos_sum**2 + sin_sum**2) / n - 1) / (n - 1)
    real_phi = cos_sum / n
    gaussian = torch.exp(-(nodes**2) / 2)
    integrand = abs_phi_sq - 2 * gaussian * real_phi + gaussian**2
    return integrand @ weights


def cf_distance_population_laplace(nodes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Exact ``NG`` of a unit-variance Laplace variable, for tests."""
    phi = 1 / (1 + nodes**2 / 2)
    gaussian = torch.exp(-(nodes**2) / 2)
    return ((phi - gaussian) ** 2) @ weights


def ey_penalty_and_reward(
    z_a: list[torch.Tensor], z_b: list[torch.Tensor] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(2 tr C, tr(V V'))`` of two-view embeddings.

    ``C`` and ``V`` come from batch ``a``; ``V'`` from the independent batch
    ``b``, or from ``a`` itself when ``b`` is None (the biased full-batch form).
    """

    def moments(z: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        zc = [e - e.mean(0) for e in z]
        n = zc[0].shape[0]
        total = sum(zc)
        c = total.T @ total / (n - 1) / len(zc)
        v = sum(e.T @ e for e in zc) / (n - 1) / len(zc)
        return c, v

    c, v = moments(z_a)
    v_ind = v if z_b is None else moments(z_b)[1]
    return 2 * torch.trace(c), torch.trace(v @ v_ind)


@dataclass(frozen=True)
class SecondMoments:
    """Fixed full-data second moments of the whitened views."""

    cross: torch.Tensor  # cov(x_1, x_2), (p, p)

    @classmethod
    def of(cls, whitened: list[np.ndarray]) -> SecondMoments:
        """From whitened, centred views."""
        n = whitened[0].shape[0]
        return cls(torch.tensor(whitened[0].T @ whitened[1] / (n - 1)))


def shared_std(w: list[torch.Tensor], moments: SecondMoments, variate: Variate) -> torch.Tensor:
    """Population std of each standardised variate, shape (k,) or (2, k).

    Views are whitened, so ``var(x_v' w) = |w|^2`` and the average variate
    ``(z_1 + z_2) / 2`` has variance ``(|w_1|^2 + |w_2|^2 + 2 w_1' S_12 w_2) / 4``.
    """
    if variate == "views":
        return torch.stack([(e**2).sum(0).sqrt() for e in w])
    var = ((w[0] ** 2).sum(0) + (w[1] ** 2).sum(0) + 2 * (w[0] * (moments.cross @ w[1])).sum(0)) / 4
    return var.sqrt()


def non_gaussianity(
    batch: list[torch.Tensor],
    w: list[torch.Tensor],
    moments: SecondMoments,
    variate: Variate,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """``sum_i NG`` of the standardised variates of one batch of whitened views."""
    std = shared_std(w, moments, variate)
    z = [x @ e for x, e in zip(batch, w)]
    if variate == "views":
        return sum(cf_distance(z[v] / std[v], nodes, weights).sum() for v in range(2)) / 2
    return cf_distance((z[0] + z[1]) / 2 / std, nodes, weights).sum()


def identifiable_ey_loss(
    batch: list[torch.Tensor],
    w: list[torch.Tensor],
    moments: SecondMoments,
    *,
    mu: float,
    variate: Variate,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    independent: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """``L_EY - mu * sum_i NG`` of whitened, centred views, with its terms."""
    reward, penalty = ey_penalty_and_reward(
        [x @ e for x, e in zip(batch, w)],
        None if independent is None else [x @ e for x, e in zip(independent, w)],
    )
    ng = non_gaussianity(batch, w, moments, variate, nodes, weights)
    return {"objective": -reward + penalty - mu * ng, "ey": -reward + penalty, "ng": ng}
