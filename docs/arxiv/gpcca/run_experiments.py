"""Experiments for the GaussianProcessCCA arxiv paper.

Produces two figures:
  fig1_interaction.pdf  -- held-out canonical correlation on a genuine
                           feature-interaction task, GaussianProcessCCA vs
                           rCCA / GAMCCA / TreeCCA, over several seeds.
  fig2_scaling.pdf      -- sparse (inducing-point) vs exact GaussianProcessCCA:
                           held-out correlation and fit time vs n_inducing,
                           at a training size that makes exact inference slow.

Also writes results_interaction.json / results_scaling.json with the raw
numbers quoted in the paper text.
"""

from __future__ import annotations

import json
import time

import numpy as np

from cca_zoo.gam import GAMCCA
from cca_zoo.gp import GaussianProcessCCA
from cca_zoo.linear import rCCA
from cca_zoo.tree import TreeCCA

OUT = "."


def interaction_data(seed: int, n_train: int = 300, n_test: int = 300, noise: float = 0.2):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    interaction = u * v
    X1 = np.column_stack([u, v]) + noise * rng.standard_normal((n, 2))
    X2 = np.column_stack([interaction, interaction]) + noise * rng.standard_normal((n, 2))
    return X1[:n_train], X1[n_train:], X2[:n_train], X2[n_train:]


def run_interaction_experiment(seeds=range(10)):
    rows = []
    for seed in seeds:
        X1_tr, X1_te, X2_tr, X2_te = interaction_data(seed)

        gp = GaussianProcessCCA(latent_dimensions=1, random_state=0)
        gp_test = gp.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

        gam = GAMCCA(latent_dimensions=1, random_state=0)
        gam_test = gam.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

        tree = TreeCCA(latent_dimensions=1, n_estimators=150, max_depth=5, random_state=0)
        tree_test = tree.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

        rcca = rCCA(latent_dimensions=1, c=[0.3, 0.3])
        rcca_test = rcca.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

        rows.append(
            dict(seed=seed, rCCA=rcca_test, GAMCCA=gam_test, TreeCCA=tree_test,
                 GaussianProcessCCA=gp_test)
        )
        print(seed, rows[-1])

    with open(f"{OUT}/results_interaction.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_scaling_experiment(n_train=4000, n_test=500, noise=0.3,
                            inducing_grid=(20, 50, 100, 200, 400)):
    rng = np.random.default_rng(0)
    n = n_train + n_test
    z = rng.standard_normal(n)
    X1 = np.column_stack([z + noise * rng.standard_normal(n) for _ in range(3)])
    X2 = np.column_stack([z**2 + noise * rng.standard_normal(n) for _ in range(3)])
    X1_tr, X1_te = X1[:n_train], X1[n_train:]
    X2_tr, X2_te = X2[:n_train], X2[n_train:]

    rows = []
    for m in inducing_grid:
        t0 = time.perf_counter()
        model = GaussianProcessCCA(latent_dimensions=1, random_state=0, n_inducing=m)
        model.fit([X1_tr, X2_tr])
        fit_time = time.perf_counter() - t0
        test_corr = model.score([X1_te, X2_te])[0]
        rows.append(dict(n_inducing=m, fit_time=fit_time, test_corr=test_corr))
        print(rows[-1])

    # exact inference reference on a smaller subsample (full n_train is too
    # slow for a "short paper" experiment budget: O(n^3))
    n_exact = 800
    t0 = time.perf_counter()
    model = GaussianProcessCCA(latent_dimensions=1, random_state=0)
    model.fit([X1_tr[:n_exact], X2_tr[:n_exact]])
    exact_time_800 = time.perf_counter() - t0
    exact_corr_800 = model.score([X1_te, X2_te])[0]

    with open(f"{OUT}/results_scaling.json", "w") as f:
        json.dump(dict(sparse=rows, exact_n800=dict(
            n_train=n_exact, fit_time=exact_time_800, test_corr=exact_corr_800)), f, indent=2)
    return rows, dict(n_train=n_exact, fit_time=exact_time_800, test_corr=exact_corr_800)


if __name__ == "__main__":
    print("=== interaction experiment ===")
    run_interaction_experiment()
    print("=== scaling experiment ===")
    run_scaling_experiment()
