from typing import Iterable

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel


class ProbabilisticCCA(BaseModel):
    """
    A class used to fit a Probabilistic CCA model using variational inference.

    Probabilistic CCA is a generative model that assumes each view of data is generated from a shared latent variable z and some view-specific parameters (mu: mean, psi: covariance, W: weight matrix). The model can be written as:

    z ~ N(0, I)
    x_i ~ N(W_i z + mu_i, psi_i)

    The model parameters and the latent variables are inferred using MCMC sampling with the NUTS algorithm.

    Parameters
    ----------
    latent_dimensions : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state, by default 0
    num_samples : int, optional
        Number of samples to use in MCMC, by default 100
    num_warmup : int, optional
        Number of warmup samples to use in MCMC, by default 100


    References
    ----------
    Bach, Francis R., and Michael I. Jordan. "A probabilistic interpretation of canonical correlation analysis." (2005).
    Wang, Chong. "Variational Bayesian approach to canonical correlation analysis." IEEE Transactions on Neural Networks 18.3 (2007): 905-910.

    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state: int = 0,
        num_samples=100,
        num_warmup=100,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            accept_sparse=False,
            random_state=random_state,
        )
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.rng_key = PRNGKey(random_state)

    def fit(self, views: Iterable[np.ndarray], y=None):
        """
        Infer the parameters and latent variables of the Probabilistic CCA model.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            A list or tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        views = self._validate_data(views)
        # Initialize a NUTS sampler with the model function
        nuts_kernel = NUTS(self._model)
        # Run MCMC sampling with the specified number of samples and warmup steps
        self.mcmc = MCMC(
            nuts_kernel, num_samples=self.num_samples, num_warmup=self.num_warmup
        )
        # Run the sampler on the data and store the posterior samples
        self.mcmc.run(self.rng_key, views)
        self.posterior_samples = self.mcmc.get_samples()
        return self

    def transform(self, views: Iterable[np.ndarray], y=None):
        """
        Predict the latent variables that generate the data in views using the sampled model parameters.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            A list or tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        z : np.ndarray
            An array of shape (n_samples, latent_dimensions) containing the predicted latent variables for each sample.
        """
        # Check if the model has been fitted
        check_is_fitted(self, attributes=["posterior_samples"])
        # Use the predictive function to generate samples of z from the posterior distribution
        return Predictive(self._model, self.posterior_samples, return_sites=["z"])(
            self.rng_key, views
        )["z"]

    def _model(self, views: Iterable[np.ndarray]):
        n = views[0].shape[0]
        p = [view.shape[1] for view in views]
        # parameter representing the mean of column in each view of data
        mu = [
            numpyro.sample(
                "mu_" + str(i), dist.MultivariateNormal(0.0, 10 * jnp.eye(p_))
            )
            for i, p_ in enumerate(p)
        ]
        # parameter representing the within view variance for each view of data
        psi = [
            numpyro.sample("psi_" + str(i), dist.LKJCholesky(p_))
            for i, p_ in enumerate(p)
        ]
        # parameter representing weights applied to latent variables
        with numpyro.plate("plate_views", self.latent_dimensions):
            self.weights_list = [
                numpyro.sample(
                    "W_" + str(i),
                    dist.MultivariateNormal(0.0, 10 * jnp.diag(jnp.ones(p_))),
                )
                for i, p_ in enumerate(p)
            ]
        with numpyro.plate("plate_i", n):
            # sample from latent z: the latent variables of the model
            z = numpyro.sample(
                "z",
                dist.MultivariateNormal(
                    0.0, jnp.diag(jnp.ones(self.latent_dimensions))
                ),
            )
            # sample from multivariate normal and observe data
            [
                numpyro.sample(
                    "obs" + str(i),
                    dist.MultivariateNormal((z @ W_) + mu_, scale_tril=psi_),
                    obs=X_,
                )
                for i, (X_, psi_, mu_, W_) in enumerate(
                    zip(views, psi, mu, self.weights_list)
                )
            ]

    def _more_tags(self):
        return {"probabilistic": True}
