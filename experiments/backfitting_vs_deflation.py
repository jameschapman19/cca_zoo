"""Backfitting vs. greedy deflation for multi-component sparse PLS/CCA.

Both fit the same penalised rank-K model of the cross-correlation matrix C of
standardised views (the diagonal-covariance approximation ParkhomenkoCCA and
PMDCCA use),

    F(A, B) = 1/2 ||C - sum_k a_k b_k'||_F^2 + lam * sum_k (|a_k|_1 + |b_k|_1).

Deflation fits component k to C minus components 1..k-1 and never revisits
it. Backfitting then keeps sweeping, refitting each component to the partial
residual C - sum_{j != k} a_j b_j' given all the others. Sweep 0 of
backfitting *is* deflation, so every sweep after is pure improvement.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

SOFT = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0.0)  # noqa: E731


def objective(C, A, B, lam):
    resid = C - A @ B.T
    return 0.5 * np.sum(resid**2) + lam * (np.abs(A).sum() + np.abs(B).sum())


def update_component(R, a, b, lam):
    """One block-coordinate step on the rank-1 fit of R: exact in a, then in b."""
    a = SOFT(R @ b, lam) / max(b @ b, 1e-12)
    b = SOFT(R.T @ a, lam) / max(a @ a, 1e-12)
    return a, b


def deflation(C, n_components, lam, inner=200, tol=1e-9):
    A = np.zeros((C.shape[0], n_components))
    B = np.zeros((C.shape[1], n_components))
    for k in range(n_components):
        R = C - A @ B.T
        u, s, vt = np.linalg.svd(R, full_matrices=False)
        a, b = np.sqrt(s[0]) * u[:, 0], np.sqrt(s[0]) * vt[0]
        for _ in range(inner):
            a_new, b_new = update_component(R, a, b, lam)
            done = max(np.abs(a_new - a).max(), np.abs(b_new - b).max()) < tol
            a, b = a_new, b_new
            if done:
                break
        A[:, k], B[:, k] = a, b
    return A, B


def backfitting(C, A, B, lam, sweeps):
    """Sweeps from the deflation solution; returns the objective after each."""
    A, B = A.copy(), B.copy()
    history = [objective(C, A, B, lam)]
    for _ in range(sweeps):
        for k in range(A.shape[1]):
            R = C - A @ B.T + np.outer(A[:, k], B[:, k])
            if not A[:, k].any():
                # A zeroed component is a fixed point of the update; re-seed it
                # from the partial residual, as deflation seeds each component.
                u, sv, vt = np.linalg.svd(R, full_matrices=False)
                A[:, k], B[:, k] = np.sqrt(sv[0]) * u[:, 0], np.sqrt(sv[0]) * vt[0]
            A[:, k], B[:, k] = update_component(R, A[:, k], B[:, k], lam)
        history.append(objective(C, A, B, lam))
    return A, B, history


def simulate(overlap, n=300, p=100, q=80, k=3, support=12, noise=1.0, seed=0, signal=0.8):
    """K latent components with sparse loadings; ``overlap`` is the share of
    each component's support shared with the previous component."""
    rng = np.random.default_rng(seed)

    def loadings(d):
        W = np.zeros((d, k))
        W[rng.choice(d, support, replace=False), 0] = rng.choice([-1, 1], support)
        for j in range(1, k):
            shared = int(round(overlap * support))
            idx = np.flatnonzero(W[:, j - 1])[:shared]
            rest = rng.choice(np.setdiff1d(np.arange(d), idx), support - shared, replace=False)
            W[np.concatenate([idx, rest]), j] = rng.choice([-1, 1], support)
        return W

    Wx, Wy = loadings(p), loadings(q)
    # Distinct strengths; scores are correlated across components via the shared
    # support, which is what makes greedy deflation misattribute signal.
    strengths = np.array([1.0, 0.8, 0.6])[:k]
    Z = rng.standard_normal((n, k)) * strengths
    X = Z @ Wx.T * signal + noise * rng.standard_normal((n, p))
    Y = Z @ Wy.T * signal + noise * rng.standard_normal((n, q))
    X, Y = (M - M.mean(0) for M in (X, Y))
    X, Y = X / X.std(0), Y / Y.std(0)
    return X.T @ Y / n, Wx, Wy


def support_f1(est, true):
    """F1 of the recovered support, matching components greedily by overlap."""
    f1s = []
    used = set()
    for j in range(true.shape[1]):
        best = (-1, 0.0)
        for i in range(est.shape[1]):
            if i in used:
                continue
            tp = np.sum((est[:, i] != 0) & (true[:, j] != 0))
            f = 2 * tp / max((est[:, i] != 0).sum() + (true[:, j] != 0).sum(), 1)
            if f > best[1]:
                best = (i, f)
        used.add(best[0])
        f1s.append(best[1])
    return float(np.mean(f1s))


def main():
    k, lam, sweeps = 3, 0.15, 40
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})

    # Sanity: at lam=0 greedy deflation is the truncated SVD (Eckart-Young), already optimal.
    C0, _, _ = simulate(0.0, noise=1.0, seed=1)
    A, B = deflation(C0, k, 0.0)
    _, _, h = backfitting(C0, A, B, 0.0, 5)
    assert abs(h[0] - h[-1]) < 1e-8, "lam=0 should leave nothing for backfitting"

    # Plot 1: objective gap vs sweep, for several overlaps.
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#4C78A8", "#F58518", "#E45756"]
    for overlap, colour in zip([0.0, 0.5, 0.9], colours):
        C, Wx, Wy = simulate(overlap)
        A, B = deflation(C, k, lam)
        _, _, h = backfitting(C, A, B, lam, 400)
        h = np.array(h)
        ax.plot(h[:sweeps + 1] - h[-1], color=colour, lw=2.5, label=f"support overlap {overlap:.0%}")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("backfitting sweep (0 = deflation)")
    ax.set_ylabel("objective − converged objective")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("experiments/backfit_convergence.png", dpi=200)

    # Plot 2: relative improvement and support recovery vs overlap.
    overlaps = np.linspace(0, 1, 6)
    rel, f1_d, f1_b = [], [], []
    for overlap in overlaps:
        r, fd, fb = [], [], []
        for seed in range(10):
            C, Wx, Wy = simulate(overlap, seed=seed)
            A, B = deflation(C, k, lam)
            Ab, Bb, h = backfitting(C, A, B, lam, 400)
            r.append((h[0] - h[-1]) / h[0])
            fd.append(support_f1(A, Wx))
            fb.append(support_f1(Ab, Wx))
        rel.append(r), f1_d.append(fd), f1_b.append(fb)
    rel, f1_d, f1_b = map(np.array, (rel, f1_d, f1_b))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.errorbar(overlaps, 100 * rel.mean(1), 100 * rel.std(1) / np.sqrt(10), color="#4C78A8", lw=2.5, marker="o", capsize=3)
    ax1.set_xlabel("support overlap between components")
    ax1.set_ylabel("objective reduction from backfitting (%)")
    for f1, colour, name in [(f1_d, "#999999", "deflation"), (f1_b, "#4C78A8", "backfitting")]:
        ax2.errorbar(overlaps, f1.mean(1), f1.std(1) / np.sqrt(10), color=colour, lw=2.5, marker="o", capsize=3, label=name)
    ax2.set_xlabel("support overlap between components")
    ax2.set_ylabel("support recovery (F1, view 1)")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("experiments/backfit_vs_overlap.png", dpi=200)

    for o, r, fd, fb in zip(overlaps, rel, f1_d, f1_b):
        print(f"overlap {o:.1f}: objective -{100 * r.mean():.2f}%  F1 {fd.mean():.3f} -> {fb.mean():.3f}")


if __name__ == "__main__":
    main()
