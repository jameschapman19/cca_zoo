"""Backfitting vs. greedy deflation for multi-component sparse PCA.

Both minimise the same penalised rank-K sparse SVD of the centred data X,

    F(U, V) = 1/2 ||X - sum_k u_k v_k'||_F^2 + lam * sum_k |v_k|_1,   ||u_k|| = 1,

the formulation behind Shen & Huang's sPCA-rSVD and Witten's PMD, where
greedy deflation is the standard way to get more than one component.
Deflation fits component k to X minus components 1..k-1 and never revisits it;
backfitting then refits each component to X minus all the others. Sweep 0 of
backfitting is deflation.

Data: X = Z diag(theta) V' + noise with orthonormal sparse V. Two hard
regimes for greedy deflation: components sharing support (orthogonal through
signs, built from Hadamard blocks), and nearly equal spikes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np



def hadamard(n):
    """Sylvester Hadamard matrix of order n, a power of two."""
    H = np.ones((1, 1))
    while len(H) < n:
        H = np.block([[H, H], [H, -H]])
    return H


SOFT = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0.0)  # noqa: E731


def objective(X, U, V, lam):
    return 0.5 * np.sum((X - U @ V.T) ** 2) + lam * np.abs(V).sum()


def fit_component(R, u, v, lam, inner=1, tol=0.0, others=None):
    """Exact block updates: v by soft-thresholding, u by the unit-norm maximiser.

    With ``others`` (orthonormal columns), u is also kept orthogonal to them.
    """
    for _ in range(inner):
        v_new = SOFT(R.T @ u, lam)
        w = R @ v_new
        if others is not None and others.size:
            w = w - others @ (others.T @ w)
        norm = np.linalg.norm(w)
        u_new = w / norm if norm > 1e-12 else u
        done = tol > 0 and np.abs(v_new - v).max() < tol
        u, v = u_new, v_new
        if done:
            break
    return u, v


def seed(R):
    u, s, vt = np.linalg.svd(R, full_matrices=False)
    return u[:, 0], s[0] * vt[0]


def deflation(X, k, lam, orth=False):
    U = np.zeros((X.shape[0], k))
    V = np.zeros((X.shape[1], k))
    for j in range(k):
        R = X - U @ V.T
        u, v = seed(R)
        others = U[:, :j] if orth else None
        U[:, j], V[:, j] = fit_component(R, u, v, lam, inner=500, tol=1e-9, others=others)
    return U, V


def backfitting(X, U, V, lam, sweeps, orth=False):
    U, V = U.copy(), V.copy()
    history = [objective(X, U, V, lam)]
    for _ in range(sweeps):
        for j in range(U.shape[1]):
            R = X - U @ V.T + np.outer(U[:, j], V[:, j])
            u, v = (U[:, j], V[:, j]) if V[:, j].any() else seed(R)
            others = np.delete(U, j, axis=1) if orth else None
            U[:, j], V[:, j] = fit_component(R, u, v, lam, others=others)
        history.append(objective(X, U, V, lam))
    return U, V, history


def true_loadings(p, k, support, shared):
    """Orthonormal sparse loadings; ``shared`` coordinates are common to all
    components, with signs from a Hadamard block so the components stay orthogonal."""
    V = np.zeros((p, k))
    if shared:
        H = hadamard(shared)  # shared in {4, 8}; columns 1..k are orthogonal
        V[:shared, :] = H[:, 1 : k + 1]
    private = support - shared
    for j in range(k):
        start = shared + j * private
        V[start : start + private, j] = np.where(np.arange(private) % 2, 1.0, -1.0) * (
            1 if j % 2 else -1
        )
    return V / np.linalg.norm(V, axis=0)


def simulate(shared, theta, n=200, p=200, support=12, seed_=0):
    rng = np.random.default_rng(seed_)
    V = true_loadings(p, len(theta), support, shared)
    Z = rng.standard_normal((n, len(theta)))
    X = Z * theta @ V.T + rng.standard_normal((n, p))
    return X - X.mean(0), V


def support_f1(est, true):
    scores, used = [], set()
    for j in range(true.shape[1]):
        best = (None, 0.0)
        for i in range(est.shape[1]):
            if i in used:
                continue
            tp = np.sum((est[:, i] != 0) & (true[:, j] != 0))
            f = 2 * tp / max((est[:, i] != 0).sum() + (true[:, j] != 0).sum(), 1)
            if f > best[1]:
                best = (i, f)
        used.add(best[0])
        scores.append(best[1])
    return float(np.mean(scores))


def subspace_error(est, true):
    """Frobenius norm of sin(principal angles); 0 for the same span."""
    if not est.any():
        return float(np.sqrt(true.shape[1]))
    q, _ = np.linalg.qr(est)
    return float(np.linalg.norm(true - q @ (q.T @ true)))


def run(shared, theta, lam, seeds=20, sweeps=300, orth=False):
    out = {"rel": [], "f1_d": [], "f1_b": [], "sub_d": [], "sub_b": []}
    for seed_ in range(seeds):
        X, V = simulate(shared, theta, seed_=seed_)
        U, Vd = deflation(X, len(theta), lam, orth)
        _, Vb, h = backfitting(X, U, Vd, lam, sweeps, orth)
        out["rel"].append((h[0] - h[-1]) / h[0])
        out["f1_d"].append(support_f1(Vd, V))
        out["f1_b"].append(support_f1(Vb, V))
        out["sub_d"].append(subspace_error(Vd, V))
        out["sub_b"].append(subspace_error(Vb, V))
    return {k: np.array(v) for k, v in out.items()}


def band(ax, x, ys, colour, label, ls="-"):
    m = np.array([y.mean() for y in ys])
    se = np.array([y.std() / np.sqrt(len(y)) for y in ys])
    ax.errorbar(x, m, se, color=colour, ls=ls, lw=2.5, marker="o", capsize=3, label=label)


def figure(x, free, orth, xlabel, path):
    """Objective gain and subspace error for backfitting, with and without an
    orthonormal-U constraint; deflation is the baseline."""
    grey, blue, orange = "#999999", "#4C78A8", "#F58518"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    band(a1, x, [r["rel"] * 100 for r in free], blue, "free U")
    band(a1, x, [r["rel"] * 100 for r in orth], orange, "orthonormal U")
    a1.set_ylabel("objective reduction from backfitting (%)")
    band(a2, x, [r["sub_d"] for r in free], grey, "deflation (free U)")
    band(a2, x, [r["sub_b"] for r in free], blue, "backfitting (free U)")
    band(a2, x, [r["sub_b"] for r in orth], orange, "orthonormal U (deflation = backfitting)")
    a2.set_ylabel("subspace error")
    a2.legend(frameon=False, fontsize=11)
    for ax in (a1, a2):
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def report(label, xs, free, orth):
    for x, f, o in zip(xs, free, orth):
        print(
            f"{label} {x}: free U objective -{100 * f['rel'].mean():.2f}% "
            f"F1 {f['f1_d'].mean():.2f}->{f['f1_b'].mean():.2f} "
            f"subspace {f['sub_d'].mean():.2f}->{f['sub_b'].mean():.2f} | "
            f"orthonormal U objective -{100 * o['rel'].mean():.3f}% "
            f"subspace {o['sub_d'].mean():.2f}->{o['sub_b'].mean():.2f}"
        )


def main():
    lam, theta = 3.5, np.array([3.0, 2.6, 2.2])
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})

    # Sanity: at lam=0 greedy deflation is the truncated SVD, already optimal.
    X, _ = simulate(0, theta)
    U, V = deflation(X, 3, 0.0)
    _, _, h = backfitting(X, U, V, 0.0, 5)
    assert abs(h[0] - h[-1]) < 1e-6 * h[0], "lam=0 should leave nothing for backfitting"

    # 1. Components sharing support (orthogonal via Hadamard signs).
    shared = [0, 4, 8]
    x = [m / 12 for m in shared]
    free = [run(m, theta, lam) for m in shared]
    orth = [run(m, theta, lam, orth=True) for m in shared]
    figure(x, free, orth, "shared support fraction", "experiments/spca_vs_overlap.png")
    report("shared", shared, free, orth)

    # 2. Nearly equal spikes, at the hardest overlap.
    gaps = [0.5, 0.7, 0.85, 0.95]
    spikes = [3.0 * np.array([1.0, g, g**2]) for g in gaps]
    free = [run(8, t, lam) for t in spikes]
    orth = [run(8, t, lam, orth=True) for t in spikes]
    figure(gaps, free, orth, "spike ratio θ$_{k+1}$/θ$_k$", "experiments/spca_vs_gap.png")
    report("ratio", gaps, free, orth)


if __name__ == "__main__":
    main()
