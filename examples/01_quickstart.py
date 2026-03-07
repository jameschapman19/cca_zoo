"""
Quickstart: CCA and PLS on simulated data
==========================================

Demonstrates the core CCA-Zoo workflow:

1. Generate correlated multiview data with ``JointData``.
2. Fit ``CCA`` and ``PLS`` on training data.
3. Evaluate canonical correlations on held-out test data.
4. Inspect weight matrices and canonical loadings.
"""

# %%
# Setup
# -----
import matplotlib.pyplot as plt
import numpy as np

from cca_zoo.datasets import JointData
from cca_zoo.linear import CCA, PLS

# %%
# Generate data
# -------------
# JointData draws from a linear latent variable model:
#   X_i = Z @ W_i.T + noise_i
# The same loading matrices W_i are reused across calls to sample(),
# so train and test sets share the same generative structure.

data = JointData(
    n_views=2,
    n_samples=300,
    n_features=[50, 40],
    latent_dimensions=3,
    signal_to_noise=2.0,
    random_state=0,
)
train_views = data.sample()
test_views = data.sample()

print("View shapes:", [v.shape for v in train_views])

# %%
# Fit CCA
# -------
cca = CCA(latent_dimensions=3).fit(train_views)

# Canonical correlations on test data, one value per latent dimension
corrs = cca.score(test_views)
print("CCA canonical correlations:", np.round(corrs, 3))

# %%
# Fit PLS
# -------
pls = PLS(latent_dimensions=3).fit(train_views)
print("PLS canonical correlations:", np.round(pls.score(test_views), 3))

# %%
# Transform: project views into the shared latent space
# -----------------------------------------------------
z_cca = cca.transform(test_views)  # list of two arrays, each (300, 3)
z1, z2 = z_cca
print("Latent shape:", z1.shape)

# %%
# Inspect weights
# ---------------
W1, W2 = cca.weights  # list of weight matrices
print("Weight shapes:", W1.shape, W2.shape)  # (50, 3), (40, 3)

# %%
# Compare CCA vs PLS correlations
# --------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
dims = np.arange(1, 4)
ax.plot(dims, cca.score(test_views), "o-", label="CCA")
ax.plot(dims, pls.score(test_views), "s-", label="PLS")
ax.set_xlabel("Latent dimension")
ax.set_ylabel("Canonical correlation")
ax.set_title("CCA vs PLS — test-set canonical correlations")
ax.legend()
ax.set_xticks(dims)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
