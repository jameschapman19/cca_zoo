"""
Multiview CCA: MCCA, GCCA, and TCCA
=====================================

Demonstrates multiview CCA methods for three or more views.

- ``MCCA`` — maximises the sum of pairwise correlations.
- ``GCCA`` — finds a shared latent projection for all views.
- ``TCCA`` — captures higher-order (tensor) correlations.
"""

# %%
# Setup
# -----
import numpy as np
import matplotlib.pyplot as plt

from cca_zoo.datasets import JointData
from cca_zoo.linear import GCCA, MCCA, TCCA

# %%
# Generate three-view data
# ------------------------
data = JointData(
    n_views=3,
    n_samples=200,
    n_features=[30, 25, 20],
    latent_dimensions=2,
    signal_to_noise=2.0,
    random_state=0,
)
train_views = data.sample()
test_views  = data.sample()

print("View shapes:", [v.shape for v in train_views])

# %%
# MCCA — Multiset CCA
# --------------------
# Solves a generalised eigenvalue problem on the block covariance matrix.
# Parameter c adds a ridge penalty to the within-view covariances.

mcca = MCCA(latent_dimensions=2, c=0.01).fit(train_views)
print("MCCA correlations:", np.round(mcca.score(test_views), 3))

# %%
# GCCA — Generalised CCA
# -----------------------
# Finds a shared n x k representation G such that each view can be
# reconstructed from it. The view_weights parameter lets you weight views.

gcca = GCCA(latent_dimensions=2, c=0.01).fit(train_views)
print("GCCA correlations:", np.round(gcca.score(test_views), 3))

# Weighted GCCA: upweight the first view
gcca_w = GCCA(latent_dimensions=2, c=0.01, view_weights=[2.0, 1.0, 1.0]).fit(train_views)
print("Weighted GCCA:", np.round(gcca_w.score(test_views), 3))

# %%
# TCCA — Tensor CCA
# ------------------
# Builds a joint cross-moment tensor of the whitened views and decomposes
# it via PARAFAC. Captures higher-order correlations beyond pairwise.

tcca = TCCA(latent_dimensions=2, c=0.01, random_state=0).fit(train_views)
print("TCCA correlations:", np.round(tcca.score(test_views), 3))

# %%
# Compare methods
# ---------------
labels = ["MCCA", "GCCA", "TCCA"]
scores = [
    mcca.score(test_views),
    gcca.score(test_views),
    tcca.score(test_views),
]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2)
width = 0.25
for i, (label, score) in enumerate(zip(labels, scores)):
    ax.bar(x + i * width, score, width, label=label)

ax.set_xlabel("Latent dimension")
ax.set_ylabel("Avg. pairwise canonical correlation")
ax.set_title("Three-view CCA methods — test-set correlations")
ax.set_xticks(x + width)
ax.set_xticklabels(["Dim 1", "Dim 2"])
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# %%
# Transform all views
# -------------------
z = mcca.transform(test_views)    # list of three arrays, each (200, 2)
print("Latent shapes:", [zi.shape for zi in z])
