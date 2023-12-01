from typing import Iterable

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import SVI

from cca_zoo._base import _BaseModel
from cca_zoo._utils._checks import check_graphviz_support


class ProbabilisticCCA(_BaseModel):
    """
    A class for performing Maximum Likelihood Estimation (MLE) in Probabilistic Canonical Correlation Analysis (CCA) using variational inference.

    Probabilistic CCA is a generative model that makes the following assumptions:

    1. A latent variable representations exists that influences both representations (X1, X2).
    2. Each observed view is generated via its own set of parameters: W (weight matrix), mu (mean), and psi (covariance).

    The generative model can be described as follows:
    representations ~ N(0, I)
    X1|representations ~ N(W1 * representations + mu1, psi1)
    X2|representations ~ N(W2 * representations + mu2, psi2)

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
        learning_rate=1e-1,
        n_iter=20000,
        num_samples=5000,
        num_warmup=5000,
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
            A list or tuple of numpy arrays representing different representations of the same samples. Each numpy array must have the same number of rows.
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
            A tuple containing the first and second representations, X1 and X2, each as a numpy array.
        """
        X1, X2 = views

        W1 = numpyro.param(
            "W_1",
            jnp.ones(
                shape=(
                    self.n_features_in_[0],
                    self.latent_dimensions,
                ),
            ),
        )
        W2 = numpyro.param(
            "W_2",
            jnp.ones(
                shape=(
                    self.n_features_in_[1],
                    self.latent_dimensions,
                ),
            ),
        )

        # Add positive-definite constraint for psi1 and psi2
        L1 = numpyro.param(
            "L_1",
            jnp.eye(self.n_features_in_[0]),
            constraint=dist.constraints.lower_cholesky,
        )
        psi1 = L1 @ L1.T
        L2 = numpyro.param(
            "L_2",
            jnp.eye(self.n_features_in_[1]),
            constraint=dist.constraints.lower_cholesky,
        )
        psi2 = L2 @ L2.T

        mu1 = numpyro.param(
            "mu_1",
            jnp.zeros(
                shape=(
                    1,
                    self.n_features_in_[0],
                ),
            ),
        )
        mu2 = numpyro.param(
            "mu_2",
            jnp.zeros(
                shape=(
                    1,
                    self.n_features_in_[1],
                ),
            ),
        )

        with numpyro.plate("n", self.n_samples_):
            z = numpyro.sample(
                "z",
                dist.MultivariateNormal(
                    jnp.zeros(self.latent_dimensions), jnp.eye(self.latent_dimensions)
                ),
            )

            numpyro.sample(
                "X1",
                dist.MultivariateNormal(
                    jnp.outer(z, W1.T) + mu1,
                    covariance_matrix=psi1,
                ),
                obs=X1,
            )
            numpyro.sample(
                "X2",
                dist.MultivariateNormal(
                    jnp.outer(z, W2.T) + mu2,
                    covariance_matrix=psi2,
                ),
                obs=X2,
            )

    def _guide(self, views):
        """
        Defines the variational distribution for Probabilistic CCA.

        Parameters
        ----------
        views: tuple of np.ndarray
            A tuple containing the first and second representations, X1 and X2, each as a numpy array.
        """

        # Variational parameters for the approximate posterior of z
        z_loc = numpyro.param(
            "z_loc", jnp.zeros((self.n_samples_, self.latent_dimensions))
        )
        z_scale = numpyro.param(
            "z_scale",
            jnp.ones((self.n_samples_, self.latent_dimensions)),
            constraint=dist.constraints.positive,
        )

        with numpyro.plate("n", self.n_samples_):
            numpyro.sample("z", dist.MultivariateNormal(z_loc, jnp.diag(z_scale)))

    def render(self, views):
        check_graphviz_support("ProbabilisticCCA")
        self.rendering = numpyro.render_model(
            self._model, model_args=(views,), filename="model.pdf"
        )

    def _more_tags(self):
        return {"probabilistic": True}

    def joint(self):
        psi1 = self.params["L_1"] @ self.params["L_1"].T
        psi2 = self.params["L_2"] @ self.params["L_2"].T
        # Calculate the individual matrix blocks
        top_left = self.params["W_1"] @ self.params["W_1"].T + psi1
        bottom_right = self.params["W_2"] @ self.params["W_2"].T + psi2
        top_right = self.params["W_1"] @ self.params["W_2"].T
        bottom_left = self.params["W_2"] @ self.params["W_1"].T

        # Construct the matrix using the blocks
        matrix = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        return matrix
