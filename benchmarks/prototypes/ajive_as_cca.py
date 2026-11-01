"""Does AJIVE's joint step reduce to a CCA method?

The claim: with the views cut to their signal subspaces, AJIVE's joint scores are
CCA's for two views (they are the normalised sum of the two canonical variates, and
the squared singular values are one plus the canonical correlations), and the
multiset CCA (GCCA) solution for more. Checked numerically, then out of sample.

    python ajive_as_cca.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "missing_views"))
from metrics import subspace_recovery  # noqa: E402
from simulate import simulate  # noqa: E402

from cca_zoo.linear import AJIVE, CCA, GCCA, IncompleteGCCA  # noqa: E402
from cca_zoo.metrics import pairwise_correlations  # noqa: E402


def principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column spaces of a and b."""
    qa = np.linalg.qr(a - a.mean(axis=0))[0]
    qb = np.linalg.qr(b - b.mean(axis=0))[0]
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def reduce(views: list[np.ndarray], rank: int) -> list[np.ndarray]:
    """Each view's top ``rank`` principal component scores."""
    return [PCA(rank).fit_transform(v) for v in views]


def two_views() -> None:
    """Two views: AJIVE's joint scores against CCA on the PCA-reduced views."""
    ds = simulate(
        0,
        n_samples=150,
        widths=(40, 60),
        n_shared=2,
        n_private=2,
        shared=0.2,
        private=0.4,
    )
    rank = 4
    ajive = AJIVE(n_components=2, signal_ranks=rank, individual_ranks=0).fit(ds.views)
    reduced = reduce(ds.views, rank)
    cca = CCA(n_components=2).fit(reduced)
    z1, z2 = cca.transform(reduced)
    canonical = np.diag(pairwise_correlations([z1, z2])[0, 1])
    # The squared singular values of the stacked orthonormal bases are 1 + rho.
    bases = [np.linalg.svd(r - r.mean(axis=0), full_matrices=False)[0] for r in reduced]
    singular = np.linalg.svd(np.concatenate(bases, axis=1), compute_uv=False)[:2]
    mean_variate = z1 + z2
    corr = [
        abs(np.corrcoef(ajive.joint_scores_[:, k], mean_variate[:, k])[0, 1])
        for k in range(2)
    ]
    print("TWO VIEWS (PCA-reduced to 4 each)")
    print(
        f"  canonical correlations of the reduced views      {np.round(canonical, 4)}"
    )
    print(
        f"  squared singular values of stacked bases - 1     {np.round(singular**2 - 1, 4)}"
    )
    print(f"  |corr(AJIVE joint score k, z1_k + z2_k)|         {np.round(corr, 4)}")


def many_views() -> None:
    """Three views: AJIVE's joint space against GCCA on the PCA-reduced views."""
    ds = simulate(
        1,
        n_samples=150,
        widths=(40, 60, 50),
        n_shared=2,
        n_private=2,
        shared=0.2,
        private=0.4,
    )
    rank = 4
    ajive = AJIVE(n_components=2, signal_ranks=rank, individual_ranks=0).fit(ds.views)
    reduced = reduce(ds.views, rank)
    gcca = GCCA(n_components=2).fit(reduced)
    latent = np.mean(gcca.transform(reduced), axis=0)
    print("THREE VIEWS (PCA-reduced to 4 each)")
    print(
        f"  principal cosines, AJIVE joint space vs GCCA     {np.round(principal_cosines(ajive.joint_scores_, latent), 4)}"
    )
    plain = GCCA(n_components=2, shrinkage=0.5).fit(ds.views)
    print(
        f"  ... vs GCCA on the full views, shrinkage 0.5     {np.round(principal_cosines(ajive.joint_scores_, np.mean(plain.transform(ds.views), axis=0)), 4)}"
    )


def out_of_sample() -> None:
    """Held-out recovery of the shared factors, and held-out correlation."""
    print("OUT OF SAMPLE (150 train, 150 test; faint shared signal, wide views)")
    print(f"  {'shared':>7} {'method':<34} {'recovery':>9} {'correlation':>12}")
    for shared in (0.02, 0.05):
        results: dict[str, list[tuple[float, float]]] = {}
        for seed in range(4):
            ds = simulate(
                seed,
                n_samples=300,
                widths=(300, 400, 200),
                n_shared=2,
                n_private=3,
                shared=shared,
                private=0.8,
            )
            train = [v[:150] for v in ds.views]
            test = [v[150:] for v in ds.views]
            truth = ds.shared_factors[150:]
            models = {
                "AJIVE (true individual ranks)": AJIVE(
                    n_components=2, individual_ranks=3
                ).fit(train),
                "AJIVE (auto ranks)": AJIVE(n_components=2).fit(train),
                "IncompleteGCCA (shrinkage 0.5)": IncompleteGCCA(
                    n_components=2, shrinkage=0.5
                ).fit(train),
                "GCCA (shrinkage 0.5)": GCCA(n_components=2, shrinkage=0.5).fit(train),
            }
            for name, model in models.items():
                latent = np.mean(model.transform(test), axis=0)
                results.setdefault(name, []).append(
                    (subspace_recovery(latent, truth), model.score(test))
                )
        for name, values in results.items():
            recovery, correlation = np.mean(values, axis=0)
            print(f"  {shared:>7} {name:<34} {recovery:>9.2f} {correlation:>12.2f}")


if __name__ == "__main__":
    two_views()
    many_views()
    out_of_sample()
