"""
Canonical Correlation Analysis for Multiview Data
==================================================

This script illustrates how to utilize the `cca_zoo` library to apply and compare
various canonical correlation analysis (CCALoss) methods for datasets with more than two representations.
"""

# %%
# Dependencies
# ------------
import numpy as np

from cca_zoo.datasets import JointData
from cca_zoo.linear import GCCA, MCCA, SCCA_PMD, TCCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA

# %%
# Data Preparation
# ----------------
# Generating a synthetic dataset with three representations (X, Y, Z) that share a common latent variable.
# Specifying the number of samples, features per view, and the latent space dimensionality.

np.random.seed(42)
n, p, q, r, latent_dims, cv = 30, 3, 3, 3, 1, 3

(X, Y, Z) = JointData(
    view_features=[p, q, r], latent_dims=latent_dims, correlation=[0.9]
).sample(n)

# %%
# Eigendecomposition-Based Methods
# --------------------------------
# These techniques leverage eigendecomposition or singular value decomposition
# to find the optimal linear transformations for the representations to maximize correlation.

# MCCALoss (Multiset CCALoss) - Generalizes CCALoss for multiple representations by maximizing pairwise correlations.
mcca = MCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# GCCALoss (Generalized CCALoss) - Maximizes correlation between each transformed view and a shared latent variable.
gcca = GCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# %%
# Kernel Methods
# --------------
# Kernel-based techniques map the original representations to a high-dimensional feature space
# and then apply linear transformations in that space.

# KCCA (Kernel CCALoss) - Kernel-based extension of CCALoss for multiple representations.
kcca = KCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# KGCCA (Kernel Generalized CCALoss) - A kernel-based version of GCCALoss for multiple representations.
kgcca = KGCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# %%
# Iterative Techniques
# --------------------
# These methods employ iterative algorithms to deduce optimal linear transformations for the representations.

# SCCA_PMD (Sparse CCALoss by Penalized Matrix Decomposition) - A sparse CCALoss variant.
pmd = (
    SCCA_PMD(latent_dimensions=latent_dims, tau=0.1, tol=1e-5)
    .fit((X, Y, X))
    .score((X, Y, Z))
)

# %%
# Tensor Decomposition Methods
# ----------------------------
# Techniques utilizing tensor decomposition to discern higher-order correlations among the representations.

# TCCALoss (Tensor CCALoss) - A tensor-based extension of CCALoss for multiple representations.
tcca = TCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# KTCCA - [Provide a brief description, as it's missing in the original].
ktcca = KTCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))
