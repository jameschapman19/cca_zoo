"""
Model selection: cross-validated hyperparameter search
=======================================================

Demonstrates ``GridSearchCV`` from ``cca_zoo.model_selection`` for
selecting hyperparameters of CCA models via cross-validation.

Topics covered:

1. Tuning the ridge penalty in ``rCCA``.
2. Tuning sparsity in ``SCCA_PMD``.
3. Comparing multiple model families.
"""

# %%
# Setup
# -----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cca_zoo.datasets import JointData
from cca_zoo.linear import CCA, SCCA_PMD, rCCA
from cca_zoo.model_selection import GridSearchCV

rng = np.random.default_rng(0)

# %%
# Generate data
# -------------
data = JointData(
    n_views=2,
    n_samples=200,
    n_features=[50, 50],
    latent_dimensions=2,
    signal_to_noise=1.5,
    random_state=0,
)
views = data.sample()
test_views = data.sample()

# %%
# Tuning rCCA: ridge penalty c
# -----------------------------
# c=0 recovers CCA, c=1 recovers PLS.
# GridSearchCV maximises the mean canonical correlation in each CV fold.

param_grid = {"c": [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]}
gs_rcca = GridSearchCV(rCCA(latent_dimensions=2), param_grid=param_grid, cv=5)
gs_rcca.fit(views)

print("rCCA — best c:", gs_rcca.best_params_["c"])
print("rCCA — best CV score:", round(gs_rcca.best_score_, 3))

# %%
# Inspect the full CV results table
# ----------------------------------
df_rcca = pd.DataFrame(gs_rcca.cv_results_)[
    ["param_c", "mean_test_score", "std_test_score"]
].sort_values("param_c")
print(df_rcca.to_string(index=False))

# %%
# Plot CV score vs c
# -------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(
    df_rcca["param_c"].astype(float),
    df_rcca["mean_test_score"],
    yerr=df_rcca["std_test_score"],
    fmt="o-",
    capsize=4,
)
ax.set_xscale("symlog", linthresh=1e-4)
ax.set_xlabel("c (ridge penalty)")
ax.set_ylabel("Mean CV canonical correlation")
ax.set_title("rCCA: cross-validated ridge penalty selection")
ax.axvline(gs_rcca.best_params_["c"], color="C1", linestyle="--", label="Best c")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Tuning SCCA_PMD: sparsity parameter τ
# ----------------------------------------
# tau controls the L1 norm bound: smaller tau → sparser weights.
# tau can be a scalar (applied to all views) or a list [tau_1, tau_2].

param_grid_pmd = {"tau": [0.1, 0.3, 0.5, 0.7, 1.0]}
gs_pmd = GridSearchCV(
    SCCA_PMD(latent_dimensions=2, random_state=0),
    param_grid=param_grid_pmd,
    cv=5,
)
gs_pmd.fit(views)

print("\nSCCA_PMD — best τ:", gs_pmd.best_params_["tau"])
print("SCCA_PMD — best CV score:", round(gs_pmd.best_score_, 3))

# %%
# Comparing CCA, rCCA (tuned), and SCCA_PMD (tuned) on test data
# ---------------------------------------------------------------
cca_score = float(CCA(latent_dimensions=2).fit(views).score(test_views).mean())
rcca_score = float(gs_rcca.best_estimator_.score(test_views).mean())
pmd_score = float(gs_pmd.best_estimator_.score(test_views).mean())

models = [
    "CCA",
    f"rCCA (c={gs_rcca.best_params_['c']})",
    f"SCCA_PMD (τ={gs_pmd.best_params_['tau']})",
]
scores = [cca_score, rcca_score, pmd_score]

fig, ax = plt.subplots(figsize=(6, 4))
colors = ["C0", "C1", "C2"]
bars = ax.bar(models, scores, color=colors, width=0.5)
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.set_ylabel("Mean test canonical correlation")
ax.set_title("Model comparison (n=200, p=50, tuned hyperparameters)")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()
