"""
Kernel CCA: nonlinear multiview analysis
=========================================

Demonstrates ``KCCA`` with several kernel choices:

- Linear kernel (recovers a kernelised version of MCCA)
- RBF (Gaussian) kernel
- Polynomial kernel
- Per-view kernel configuration
- Custom kernel function

Kernel hyperparameters are selected via ``GridSearchCV``.
"""

# %%
# Setup
# -----
import numpy as np
import matplotlib.pyplot as plt

from cca_zoo.datasets import JointData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

rng = np.random.default_rng(0)

# %%
# Generate data
# -------------
# We use a moderate-size dataset; KCCA stores an n×n kernel matrix so
# keep n ≤ a few thousand for memory efficiency.

data = JointData(
    n_views=2,
    n_samples=200,
    n_features=[20, 20],
    latent_dimensions=2,
    signal_to_noise=2.0,
    random_state=0,
)
train_views = data.sample()
test_views  = data.sample()

# %%
# Linear kernel
# -------------
# With a linear kernel KCCA is equivalent to regularised MCCA in feature space.

kcca_linear = KCCA(latent_dimensions=2, kernel="linear", c=0.1).fit(train_views)
print("Linear KCCA:", np.round(kcca_linear.score(test_views), 3))

# %%
# RBF kernel
# ----------
# The RBF kernel captures smooth nonlinear relationships.
# gamma controls the width: smaller gamma → smoother decision boundary.

kcca_rbf = KCCA(latent_dimensions=2, kernel="rbf", gamma=0.1, c=0.1).fit(train_views)
print("RBF KCCA:   ", np.round(kcca_rbf.score(test_views), 3))

# %%
# Polynomial kernel
# -----------------
kcca_poly = KCCA(latent_dimensions=2, kernel="poly", degree=2, c=0.1).fit(train_views)
print("Poly KCCA:  ", np.round(kcca_poly.score(test_views), 3))

# %%
# Per-view kernel configuration
# ------------------------------
# Pass lists to use different kernels for each view.

kcca_mixed = KCCA(
    latent_dimensions=2,
    kernel=["rbf", "linear"],
    gamma=[0.1, None],
    c=[0.1, 0.5],
).fit(train_views)
print("Mixed KCCA: ", np.round(kcca_mixed.score(test_views), 3))

# %%
# Custom kernel function
# ----------------------
# Any callable with signature f(X, Y, **params) -> np.ndarray is accepted.

def laplacian_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Laplacian (L1-RBF) kernel: exp(-gamma * ||x - y||_1)."""
    diff = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=-1)
    return np.exp(-gamma * diff)

kcca_custom = KCCA(
    latent_dimensions=2,
    kernel=laplacian_kernel,
    kernel_params={"gamma": 0.05},
    c=0.1,
).fit(train_views)
print("Laplacian KCCA:", np.round(kcca_custom.score(test_views), 3))

# %%
# Hyperparameter tuning with GridSearchCV
# ----------------------------------------
# Cross-validate over gamma and c to find the best RBF kernel parameters.

param_grid = {
    "gamma": [0.01, 0.05, 0.1, 0.5],
    "c": [0.01, 0.1, 1.0],
}
gs = GridSearchCV(
    KCCA(latent_dimensions=2, kernel="rbf"),
    param_grid=param_grid,
    cv=5,
)
gs.fit(train_views)

print("\nBest params:", gs.best_params_)
print("Best CV score:", round(gs.best_score_, 3))
print("Test score:   ", np.round(gs.best_estimator_.score(test_views), 3))

# %%
# Visualise test-set latent representations
# ------------------------------------------
best = gs.best_estimator_
z1, z2 = best.transform(test_views)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.scatter(z1[:, 0], z1[:, 1], alpha=0.6, s=20)
ax1.set_title("View 1 latent space (KCCA, best RBF)")
ax1.set_xlabel("Dim 1")
ax1.set_ylabel("Dim 2")

ax2.scatter(z1[:, 0], z2[:, 0], alpha=0.6, s=20)
ax2.set_title("Canonical variate 1: view 1 vs view 2")
ax2.set_xlabel("View 1 — dim 1")
ax2.set_ylabel("View 2 — dim 1")
corr = np.corrcoef(z1[:, 0], z2[:, 0])[0, 1]
ax2.set_title(f"Dim 1 correlation: {corr:.3f}")

plt.tight_layout()
plt.show()
