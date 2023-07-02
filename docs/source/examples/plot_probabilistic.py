"""
Probabilistic CCA Example
===================================

This example shows how to use the `ProbabilisticCCA` class to fit a probabilistic CCA model on some synthetic data. Probabilistic CCA is a generative model that assumes each view of data is generated from a latent variable and some view-specific parameters. It uses variational inference methods to estimate the posterior distributions of the parameters and the latent variables.
"""

# %%
# Imports
# -------

import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.probabilistic import ProbabilisticCCA

"""
Data
-----
"""
# %%
# We generate some synthetic data from two views, each with 10 features. We assume that the latent variable has 2 dimensions and that the data is noisy.


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

"""
Model
------
"""
# %%
# We create an instance of the `ProbabilisticCCA` class and fit it on the data. We specify the number of latent dimensions, the number of samples and warmup steps for the variational inference algorithm, and the random state.

pcca = ProbabilisticCCA(
    latent_dimensions=latent_dims,
    num_samples=250,
    num_warmup=250,
    random_state=random_state,
)

pcca.fit(views)

"""
Results
-------
"""
# %%
# We can use the `transform` method to obtain the posterior mean of the latent variable for each sample. This can be used for downstream tasks such as visualization or clustering.

z = pcca.transform(views)

# %%
# We can also inspect the results of the model by looking at the posterior samples of the parameters and the latent variable. The `posterior_samples` attribute is a dictionary that contains the samples for each parameter and the latent variable. We can use arviz to visualize the distributions and compare them with the true values.

# Convert posterior samples to arviz InferenceData object
idata = az.from_numpyro(pcca.mcmc)

for view in range(num_views):
    # Plot the posterior distribution of W_0 parameter (for just the first latent variable). Label the weights with their weight index. Make all parameters share x axis.
    trace_plot = az.plot_trace(
        idata, var_names=[f"W_{view}"], compact=False, divergences=None
    )

    # For each w in W_0, plot the true value from data.true_features[0]
    for i, ax in enumerate(trace_plot[:, 0]):
        ax.axvline(
            data.true_features[view].ravel()[i],
            color="red",
            linestyle="--",
            label="True Value",
        )
        ax.legend()

    ax_list = list(trace_plot[:, 0].ravel())
    ax_list[0].get_shared_x_axes().join(ax_list[0], *ax_list)
    plt.suptitle(f"Posterior Distribution of W_{view}")
    plt.tight_layout()

plt.show()
