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

n = 100 # number of samples
p = [10, 10] # number of features for each view
latent_dims = 2 # number of latent dimensions

data=LinearSimulatedData(
    view_features=p, latent_dims=latent_dims, correlation=[0.9, 0.7]
)

views = data.sample(n)

"""
Model
------
"""
# %%
# We create an instance of the `ProbabilisticCCA` class and fit it on the data. We specify the number of latent dimensions, the number of samples and warmup steps for the variational inference algorithm, and the random state.

pcca = ProbabilisticCCA(
    latent_dimensions=latent_dims,
    num_samples=1000,
    num_warmup=500,
    random_state=0
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

# Plot the posterior distributions of mu, psi and W parameters for each view
az.plot_forest(idata, var_names=["mu", "psi", "W"], combined=True)

# Plot the posterior distribution of z parameter (latent variable)
az.plot_density(idata, var_names=["z"])

plt.show()
