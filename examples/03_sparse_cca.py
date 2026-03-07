"""
Sparse CCA: comparing regularised variants
===========================================

Demonstrates the sparse and regularised CCA methods in ``cca_zoo.linear``.
These methods are useful when you expect the true canonical directions to be
sparse (i.e. only a small number of features drive the correlation).

Methods compared:

- ``CCA`` — baseline (dense)
- ``SCCA_PMD`` — L1 constraint via bisection (Witten 2009)
- ``SCCA_ADMM`` — L1 constraint via ADMM (Suo 2017)
- ``SCCA_IPLS`` — elastic net at each ALS step (Mai & Zhang 2019)
- ``ElasticCCA`` — elastic net on sum-of-scores target (Waaijenborg 2008)
- ``ParkhomenkoCCA`` — fixed soft-threshold (Parkhomenko 2009)
"""

# %%
# Setup
# -----
import numpy as np
import matplotlib.pyplot as plt

from cca_zoo.datasets import JointData
from cca_zoo.linear import (
    CCA,
    ElasticCCA,
    ParkhomenkoCCA,
    SCCA_ADMM,
    SCCA_IPLS,
    SCCA_PMD,
)

rng = np.random.default_rng(0)

# %%
# Generate data with sparse true structure
# -----------------------------------------
# We use a high-dimensional setting (n < p) where sparse methods are expected
# to outperform dense CCA by recovering the true support.

data = JointData(
    n_views=2,
    n_samples=150,
    n_features=[100, 100],
    latent_dimensions=1,
    signal_to_noise=3.0,
    random_state=0,
)
train_views = data.sample()
test_views  = data.sample()

# %%
# Fit models
# ----------
models = {
    "CCA":              CCA(latent_dimensions=1),
    "SCCA_PMD (τ=0.3)": SCCA_PMD(latent_dimensions=1, tau=0.3, random_state=0),
    "SCCA_ADMM (τ=0.1)": SCCA_ADMM(latent_dimensions=1, tau=0.1, random_state=0),
    "SCCA_IPLS (α=0.01)": SCCA_IPLS(latent_dimensions=1, alpha=0.01, l1_ratio=1.0, random_state=0),
    "ElasticCCA (α=0.01)": ElasticCCA(latent_dimensions=1, alpha=0.01, l1_ratio=0.5, random_state=0),
    "ParkhomenkoCCA (τ=0.05)": ParkhomenkoCCA(latent_dimensions=1, tau=0.05, random_state=0),
}

results = {}
for name, model in models.items():
    model.fit(train_views)
    results[name] = {
        "score": float(model.score(test_views)[0]),
        "nnz_v1": int(np.sum(np.abs(model.weights[0][:, 0]) > 1e-6)),
        "nnz_v2": int(np.sum(np.abs(model.weights[1][:, 0]) > 1e-6)),
    }

# %%
# Print results
# -------------
print(f"{'Method':<30} {'Test corr':>10} {'NNZ v1':>8} {'NNZ v2':>8}")
print("-" * 60)
for name, res in results.items():
    print(f"{name:<30} {res['score']:>10.3f} {res['nnz_v1']:>8} {res['nnz_v2']:>8}")

# %%
# Visualise weight sparsity
# -------------------------
fig, axes = plt.subplots(len(models), 2, figsize=(10, 2.5 * len(models)), sharex=True)
for ax_row, (name, model) in zip(axes, models.items()):
    for ax, w in zip(ax_row, model.weights):
        ax.stem(np.abs(w[:, 0]), markerfmt="C0.", linefmt="C0-", basefmt="k-")
        ax.set_ylabel("|weight|", fontsize=8)

    ax_row[0].set_title(f"{name}\n(view 1)", fontsize=9)
    ax_row[1].set_title(f"{name}\n(view 2)", fontsize=9)

axes[-1][0].set_xlabel("Feature index")
axes[-1][1].set_xlabel("Feature index")
fig.suptitle("Sparse CCA weight vectors (absolute values)", fontsize=11)
plt.tight_layout()
plt.show()

# %%
# Test correlation vs sparsity trade-off
# ---------------------------------------
tau_values = np.linspace(0.1, 1.0, 10)
pmd_scores = []
pmd_nnz    = []

for tau in tau_values:
    m = SCCA_PMD(latent_dimensions=1, tau=tau, random_state=0).fit(train_views)
    pmd_scores.append(float(m.score(test_views)[0]))
    pmd_nnz.append(int(np.sum(np.abs(m.weights[0][:, 0]) > 1e-6)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(tau_values, pmd_scores, "o-")
ax1.set_xlabel("τ (L1 bound scale)")
ax1.set_ylabel("Test canonical correlation")
ax1.set_title("SCCA_PMD: correlation vs τ")

ax2.plot(tau_values, pmd_nnz, "o-", color="C1")
ax2.set_xlabel("τ (L1 bound scale)")
ax2.set_ylabel("Non-zero features (view 1)")
ax2.set_title("SCCA_PMD: sparsity vs τ")

plt.tight_layout()
plt.show()
