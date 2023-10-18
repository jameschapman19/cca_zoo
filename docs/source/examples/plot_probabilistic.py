"""
Probabilistic Canonical Correlation Analysis (CCALoss)
==================================================
Illustrates the usage of `ProbabilisticCCA` for understanding multiview data relationships.

Overview:
---------
Probabilistic CCALoss is a generative model that captures shared information among multiple representations of data. By assuming that each data view originates from a latent variable and view-specific parameters, this model offers a more flexible representation. It employs variational inference to approximate the posterior distributions of parameters and latent variables.

Contents:
---------
1. Imports and setup.
2. Data generation: Synthetic data from two representations, considering view-specific noise and feature sparsity.
3. Model: Initialize and fit `ProbabilisticCCA` on the synthetic data.
4. Results: Extract and visualize the latent variable's posterior mean and compare inferred parameters with ground truth.

Let's dive in!
"""

# %%
# 1. Imports and Setup
# --------------------

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from cca_zoo.probabilistic import ProbabilisticCCA
from cca_zoo.visualisation import WeightInferenceDisplay

# %%
# 2. Data Generation
# ------------------

# Here, we design a helper class to simulate data from two representations. Both representations contain 10 features, and data is generated from a 2-dimensional latent variable. Noise and sparsity parameters help make the data generation process more intricate.


class LatentVariableData:
    def __init__(
        self,
        latent_dimensions,
        num_views,
        view_dimensions,
        sparsity,
        noise,
        random_state=None,
    ):
        self.latent_dimensions = latent_dimensions
        self.num_views = num_views
        self.view_dimensions = view_dimensions
        self.sparsity = sparsity
        self.noise = noise
        self.random_state = random_state
        self.true_features = self._generate_true_features()

    def _generate_true_features(self):
        rng = np.random.RandomState(self.random_state)
        features = []
        for i in range(self.num_views):
            # Generate a random vector with sparsity proportion of non-zero entries
            w = rng.randn(self.latent_dimensions, self.view_dimensions)
            mask = rng.rand(*w.shape) < self.sparsity
            w[mask] = 0
            features.append(w)
        return features

    def sample(self, n):
        rng = np.random.RandomState(self.random_state)
        z = rng.randn(n, self.latent_dimensions)
        views = []
        for i in range(self.num_views):
            view = z @ self.true_features[i]
            view += self.noise * rng.randn(n, self.view_dimensions)
            views.append(view)
        return views


n = 100
num_views = 2
latent_dims = 2
view_dims = 10
sparsity = 0.5
noise = 0.1
random_state = 0

data = LatentVariableData(
    latent_dimensions=latent_dims,
    num_views=num_views,
    view_dimensions=view_dims,
    sparsity=sparsity,
    noise=noise,
    random_state=random_state,
)

views = data.sample(n)

# remove the mean from each view
views = [view - view.mean(axis=0) for view in views]

# %%
# 3. Model
# --------

# Instantiate `ProbabilisticCCA`, specifying the latent dimension, number of samples, warm-up steps, and random seed. Subsequently, fit the model on the de-meaned data representations.

pcca = ProbabilisticCCA(
    latent_dimensions=latent_dims,
    num_samples=1,
    num_warmup=1,
    random_state=random_state,
)

pcca.fit(views)

# %%
# 4. Results
# ----------

# Explore the model's results:
# - Transform the representations to obtain the latent variable's posterior mean. Useful for visualization, clustering, etc.
# - Inspect and visualize the posterior parameter distributions, comparing them with their true values.

z = pcca.transform(views)
