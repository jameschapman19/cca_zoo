from typing import Iterable

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.random import PRNGKey
from numpyro.infer import SVI, MCMC, NUTS
from cca_zoo._base import BaseModel
from numpyro import handlers


class ProbabilisticCCA(BaseModel):
    """
    A class for performing Maximum Likelihood Estimation (MLE) in Probabilistic Canonical Correlation Analysis (CCA) using variational inference.

    Probabilistic CCA is a generative model that makes the following assumptions:

    1. A latent variable z exists that influences both views (X1, X2).
    2. Each observed view is generated via its own set of parameters: W (weight matrix), mu (mean), and psi (covariance).

    The generative model can be described as follows:
    z ~ N(0, I)
    X1|z ~ N(W1 * z + mu1, psi1)
    X2|z ~ N(W2 * z + mu2, psi2)

    Parameters
    ----------
    latent_dimensions: int, optional
        The dimensionality of the latent space, by default 1.
    copy_data: bool, optional
        Whether to copy the data, by default True.
    random_state: int, optional
        The seed for the random number generator, by default 0.
    learning_rate: float, optional
        The learning rate for the optimizer, by default 1e-3.
    n_iter: int, optional
        Number of iterations for optimization, by default 10000.
    num_samples: int, optional
        Number of MCMC samples, by default 100.

    References
    ----------
    [1] Bach, Francis R., and Michael I. Jordan. "A probabilistic interpretation of canonical correlation analysis." (2005).
    [2] Wang, Chong. "Variational Bayesian approach to canonical correlation analysis." IEEE Transactions on Neural Networks 18.3 (2007): 905-910.

    """

    return_sites = ["z"]

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state: int = 0,
        learning_rate=1e-4,
        n_iter=20000,
        num_samples=1000,
        num_warmup=500,
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
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.params = None

    def fit(self, views: Iterable[np.ndarray], y=None):
        """
        Infer the parameters and latent variables of the Probabilistic Canonical Correlation Analysis (CCA) model.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            A list or tuple of numpy arrays representing different views of the same samples. Each numpy array must have the same number of rows.
        y: Any, optional
            Ignored in this implementation.

        Returns
        -------
        self : object
            Returns the instance itself, updated with the inferred parameters and latent variables.

        Notes
        -----
        - The data in each view should be normalized for optimal performance.
        """
        views = self._validate_data(views)
        self._check_params()
        svi = SVI(
            self._model,
            self._guide,
            numpyro.optim.Adam(self.learning_rate),
            loss=numpyro.infer.Trace_ELBO(),
        )
        self.svi_result = svi.run(self.rng_key, self.n_iter, views)
        self.params = self.svi_result.params
        return self

    def _model(self, views):
        """
        Defines the generative model for Probabilistic CCA.

        Parameters
        ----------
        views: tuple of np.ndarray
            A tuple containing the first and second views, X1 and X2, each as a numpy array.
        """
        X1, X2 = views

        W1 = numpyro.param(
            "W_1",
            random.normal(
                shape=(
                    self.n_features_[0],
                    self.latent_dimensions,
                ),
                key=self.rng_key,
            ),
        )
        W2 = numpyro.param(
            "W_2",
            random.normal(
                shape=(
                    self.n_features_[1],
                    self.latent_dimensions,
                ),
                key=self.rng_key,
            ),
        )

        # Add positive-definite constraint for psi1 and psi2
        psi1 = numpyro.param("psi_1", jnp.eye(self.n_features_[0]))
        psi2 = numpyro.param("psi_2", jnp.eye(self.n_features_[1]))

        mu1 = numpyro.param(
            "mu_1",
            random.normal(
                shape=(
                    1,
                    self.n_features_[0],
                ),
                key=self.rng_key,
            ),
        )
        mu2 = numpyro.param(
            "mu_2",
            random.normal(
                shape=(
                    1,
                    self.n_features_[1],
                ),
                key=self.rng_key,
            ),
        )

        n_samples = X1.shape[0] if X1 is not None else X2.shape[0]

        with numpyro.plate("n", n_samples):
            z = numpyro.sample(
                "z",
                dist.MultivariateNormal(
                    jnp.zeros(self.latent_dimensions), jnp.eye(self.latent_dimensions)
                ),
            )

            numpyro.sample(
                "X1",
                dist.MultivariateNormal(z @ W1.T + mu1, covariance_matrix=psi1),
                obs=X1,
            )
            numpyro.sample(
                "X2",
                dist.MultivariateNormal(z @ W2.T + mu2, covariance_matrix=psi2),
                obs=X2,
            )

    def _guide(self, views):
        """
        Defines the variational family (guide) for approximate inference in Probabilistic CCA.

        Parameters
        ----------
        views: tuple of np.ndarray
            A tuple containing the first and second views, X1 and X2, each as a numpy array.
        """
        X1, X2 = views

        n = X1.shape[0] if X1 is not None else X2.shape[0]

        # # Variational parameters
        # z_loc = numpyro.param("z_loc", jnp.zeros((n, self.latent_dimensions)))
        # z_scale = numpyro.param(
        #     "z_scale",
        #     jnp.ones((n, self.latent_dimensions)),
        #     constraint=dist.constraints.positive,
        # )

        with numpyro.plate("n", n):
            numpyro.sample("z", dist.MultivariateNormal(jnp.zeros(self.latent_dimensions), jnp.eye(self.latent_dimensions)))

    def transform(self, views: Iterable[np.ndarray], y=None, return_std=False):
        """
        Transform the data into the latent space.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            A list or tuple of numpy arrays representing different views of the same samples. Each numpy array must have the same number of rows.
        y: Any, optional
            Ignored in this implementation.

        Returns
        -------
        z : np.ndarray
            The transformed data in the latent space.
        """
        conditioned_model = handlers.substitute(self._model, self.params)
        kernel = NUTS(conditioned_model)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)
        mcmc.run(self.rng_key, views)
        samples = mcmc.get_samples()
        z = samples["z"]
        if return_std:
            return np.array(z.mean(axis=0)), np.array(z.std(axis=0))
        else:
            return np.array(z.mean(axis=0))

    def render(self, views):
        # check if graphviz is installed
        try:
            import graphviz
        except ImportError:
            raise ImportError("In order to use render, graphviz must be installed.")
        self.rendering = numpyro.render_model(
            self._model, model_args=(views,), filename="model.pdf"
        )

    def _more_tags(self):
        return {"probabilistic": True}

    def joint(self):
        # Calculate the individual matrix blocks
        top_left = self.params["W_1"] @ self.params["W_1"].T + self.params["psi_1"]
        bottom_right = self.params["W_2"] @ self.params["W_2"].T + self.params["psi_2"]
        top_right = self.params["W_1"] @ self.params["W_2"].T
        bottom_left = self.params["W_2"] @ self.params["W_1"].T

        # Construct the matrix using the blocks
        matrix = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        return matrix
