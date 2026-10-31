"""Linear two-view SSL arms for the batch-size / embedding-dimension sweep.

Every arm trains linear encoders ``z_v = x_v W_v`` by minibatch Adam on a loss
built from one batch of ``n`` samples:

* ``ey_single``: the EY loss with all moments from the one batch (biased penalty).
* ``ey_split``: the batch split in halves ``a, b``; the penalty is ``tr(V_a V_b)``
  (unbiased) and the reward the mean of both halves' ``2 tr C``. Same samples.
* ``sigreg_v``: LeJEPA, SIGReg as in :mod:`cca_zoo.deep._lejepa` (diagonal kept).
* ``sigreg_u``: LeJEPA with SIGReg's ``|ecf|^2`` replaced by its off-diagonal
  U-statistic, so the statistic is unbiased for the population distance.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from cca_zoo.deep._lejepa import _sigreg
from sklearn.linear_model import Ridge

from ieye.data import TwoViewModel
from ieye.loss import ey_penalty_and_reward

LAMBDA = 0.05  # cca_zoo's LeJEPA default
N_SLICES, N_POINTS, T_MAX = 256, 17, 5.0  # cca_zoo's LeJEPA defaults


def sigreg_u(z: torch.Tensor, seed: int) -> torch.Tensor:
    """:func:`~cca_zoo.deep._lejepa._sigreg` with the unbiased ``|ecf|^2``.

    Same slices, grid and ``n`` scaling; ``|ecf|^2 = C^2 + S^2`` becomes
    ``(n (C^2 + S^2) - 1) / (n - 1)`` with ``C, S`` the mean cosine and sine.
    """
    n = z.shape[0]
    generator = torch.Generator(device=z.device)
    generator.manual_seed(seed)
    directions = torch.randn(z.shape[1], N_SLICES, generator=generator, dtype=z.dtype)
    directions = directions / directions.norm(dim=0)
    t = torch.linspace(-T_MAX, T_MAX, N_POINTS, dtype=z.dtype)
    gaussian_cf = torch.exp(-0.5 * t**2)
    xt = (z @ directions).unsqueeze(2) * t
    cos_mean, sin_mean = xt.cos().mean(0), xt.sin().mean(0)
    abs_sq = (n * (cos_mean**2 + sin_mean**2) - 1) / (n - 1)
    error = abs_sq - 2 * gaussian_cf * cos_mean + gaussian_cf**2
    return (torch.trapezoid(error * gaussian_cf, t, dim=1) * n).mean()


def _lejepa(sigreg: Callable[[torch.Tensor, int], torch.Tensor]) -> Callable:
    def loss(z: list[torch.Tensor], step: int) -> torch.Tensor:
        stacked = torch.stack(z)
        sim = (stacked.mean(0) - stacked).square().mean()
        reg = torch.stack([sigreg(e, step) for e in z]).mean()
        return (1 - LAMBDA) * sim + LAMBDA * reg

    return loss


def _ey_single(z: list[torch.Tensor], step: int) -> torch.Tensor:
    reward, penalty = ey_penalty_and_reward(z, None)
    return penalty - reward


def _ey_split(z: list[torch.Tensor], step: int) -> torch.Tensor:
    half = z[0].shape[0] // 2
    a, b = [e[:half] for e in z], [e[half : 2 * half] for e in z]
    reward_a, penalty = ey_penalty_and_reward(a, b)
    reward_b, _ = ey_penalty_and_reward(b, a)
    return penalty - (reward_a + reward_b) / 2


def _sigreg_v(z: torch.Tensor, step: int) -> torch.Tensor:
    return _sigreg(z, step, N_SLICES, N_POINTS, T_MAX)


ARMS: dict[str, Callable] = {
    "EY single-batch": _ey_single,
    "EY split-batch (unbiased)": _ey_split,
    "LeJEPA (SIGReg)": _lejepa(_sigreg_v),
    "LeJEPA (U-stat SIGReg)": _lejepa(sigreg_u),
}


def train(
    arm: str,
    views: list[np.ndarray],
    d: int,
    batch: int,
    steps: int,
    lr: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Linear encoder weights ``[W_1, W_2]`` (p, d) after ``steps`` Adam steps.

    Batches are i.i.d. draws with replacement from ``views``.
    """
    loss_fn = ARMS[arm]
    x = [torch.tensor(v, dtype=torch.float32) for v in views]
    generator = torch.Generator().manual_seed(int(rng.integers(2**31)))
    w = [(torch.randn(v.shape[1], d, generator=generator) / v.shape[1] ** 0.5).requires_grad_() for v in x]
    opt = torch.optim.Adam(w, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        idx = torch.randint(x[0].shape[0], (batch,), generator=generator)
        loss = loss_fn([xv[idx] @ wv for xv, wv in zip(x, w)], step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{arm}: non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return [e.detach().numpy().astype(np.float64) for e in w]


def subspace_capture(model: TwoViewModel, weights: list[np.ndarray]) -> float:
    """Mean over views of the fraction of the CCA subspace inside the span of ``W_v``.

    In the population-whitened metric the CCA subspace is the first ``k``
    coordinates; the span keeps singular directions above 1e-3 of the largest.
    """
    scores = []
    for v, w in enumerate(weights):
        u, s, _ = np.linalg.svd(model.mixings[v].T @ w, full_matrices=False)
        basis = u[:, s > 1e-3 * s[0]]
        scores.append(np.sum(basis[: model.k] ** 2) / model.k)
    return float(np.mean(scores))


def probe_r2(
    model: TwoViewModel,
    weights: list[np.ndarray],
    rng: np.random.Generator,
    n_probe: int = 2000,
    n_test: int = 20000,
) -> float:
    """Test R^2 of a ridge probe from view-1's embedding to ``y = sum_i s_i + noise``."""
    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        s, views = model.sample(n, rng)
        return views[0] @ weights[0], s.sum(1) + 0.5 * rng.standard_normal(n)

    z_train, y_train = draw(n_probe)
    z_test, y_test = draw(n_test)
    mean, std = z_train.mean(0), z_train.std(0) + 1e-12
    probe = Ridge(alpha=1.0).fit((z_train - mean) / std, y_train)
    return float(probe.score((z_test - mean) / std, y_test))
