"""
More than 2 views
===========================

This example demonstrates how to use cca_zoo to compare different methods of canonical correlation analysis
(CCA) and related methods for more than two views of data.
"""

# %%
# Imports
# -------

import numpy as np
from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.linear import GCCA, MCCA, SCCA_PMD, TCCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA

"""
Data
-----
"""
# %%
# We generate some synthetic data with three views (views, Y, Z) that share a common latent variable.
# We set the number of samples (n), the number of features per view (p, q, r), and the dimensionality of the latent space (latent_dims).
# We also set the correlation between the views and the latent variable to 0.9.

np.random.seed(42)
n = 30
p = 3
q = 3
r = 3
latent_dims = 1
cv = 3

(X, Y, Z) = LinearSimulatedData(
    view_features=[p, q, r], latent_dims=latent_dims, correlation=[0.9]
).sample(n)

# %%
# Eigendecomposition-Based Methods
# ---------------------------------
# These methods use eigendecomposition or singular value decomposition to find the optimal linear transformations
# of the views that maximize their correlation.

# %%
# Linear
# ^^^^^^^^
# These methods use linear transformations of the original views.

# %%
# MCCA (Multiset CCA) is a generalization of CCA for more than two views.
# It maximizes the sum of pairwise correlations between the transformed views.

mcca = MCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# %%
# GCCA (Generalized CCA) is another generalization of CCA for more than two views.
# It maximizes the correlation between each transformed view and a common latent variable.

gcca = GCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# %%
# Kernel
# ^^^^^^^^
# These methods use kernel functions to map the original views to a higher-dimensional feature space,
# and then apply linear methods in that space.

# %%
# KCCA (Kernel CCA) is a kernel-based extension of CCA for two or more views.
# It maximizes the sum of pairwise kernel correlations between the transformed views.

kcca = KCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# %%
# KGCCA (Kernel Generalized CCA) is a kernel-based extension of GCCA for two or more views.
# It maximizes the kernel correlation between each transformed view and a common latent variable.

kgcca = KGCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# %%
# Iterative Methods
# ^^^^^^^^^^^^^^^^^^
# These methods use iterative algorithms to find the optimal linear transformations of the views.

# Most of the _iterative methods can also use multiple views e.g.

# %%
# SCCA_PMD (Sparse CCA by Penalized Matrix Decomposition) is a sparse variant of CCA for two or more views.
# It uses an alternating optimization algorithm with L1-norm regularization to find sparse solutions.

pmd = (
    SCCA_PMD(latent_dimensions=latent_dims, tau=0.1, tol=1e-5)
    .fit((X, Y, X))
    .score((X, Y, Z))
)

# %%
# Higher Order Correlations
# -------------------------
# These methods use tensor decomposition to find higher order correlations among the views.

# %%
# TCCA (Tensor CCA) is a tensor-based extension of CCA for two or more views.
# It finds the optimal linear transformations that maximize the higher order correlation tensor among the views.
tcca = TCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))

# %%
ktcca = KTCCA(latent_dimensions=latent_dims).fit((X, Y, X)).score((X, Y, Z))
