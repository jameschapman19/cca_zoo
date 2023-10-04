import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random

from cca_zoo.probabilistic._cca import ProbabilisticCCA
import numpy as np

from cca_zoo.utils import _process_parameter


class ProbabilisticRCCA(ProbabilisticCCA):
    """
    Probabilistic Ridge Canonical Correlation Analysis (Probabilistic Ridge CCA).

    Probabilistic Ridge CCA extends the Probabilistic Canonical Correlation Analysis model
    by introducing regularization terms in the linear relationships between multiple views
    of data. This regularization improves the conditioning of the problem and provides a
    way to incorporate prior knowledge. It combines features of both CCA and Ridge Regression.

    Parameters
    ----------
    latent_dimensions: int, default=2
        Number of latent dimensions.

    c: float, default=1.0
        Regularization strength; must be a positive float. Regularization improves
        the conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization.

    learning_rate: float, default=0.01
        Learning rate for optimization algorithms.

    n_iter: int, default=1000
        Number of iterations for optimization algorithms.

    Attributes
    ----------
    params : dict
        A dictionary containing the parameters of the fitted model.

    svi_result : object
        An object that stores results from Stochastic Variational Inference.

    References
    ----------

    [1] De Bie, T. and De Moor, B., 2003. On the regularization of canonical correlation analysis. Int. Sympos. ICA and BSS, pp.785-790.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state: int = 0,
        learning_rate=1e-4,
        n_iter=20000,
        num_samples=1000,
        num_warmup=500,
        c=1.0,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            learning_rate=learning_rate,
            n_iter=n_iter,
            num_samples=num_samples,
            num_warmup=num_warmup,
        )
        self.c = c

    def _check_params(self):
        # Process the c parameter for each view
        self.c = _process_parameter("c", self.c, 0, self.n_views_)

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
                dist.MultivariateNormal(
                    z @ W1.T + mu1,
                    covariance_matrix=psi1 + self.c[0] * jnp.eye(self.n_features_[0]),
                ),
                obs=X1,
            )
            numpyro.sample(
                "X2",
                dist.MultivariateNormal(
                    z @ W2.T + mu2,
                    covariance_matrix=psi2 + self.c[1] * jnp.eye(self.n_features_[1]),
                ),
                obs=X2,
            )

    def joint(self):
        # Calculate the individual matrix blocks
        top_left = self.params["W_1"] @ self.params["W_1"].T + self.c[0] * jnp.eye(
            self.n_features_[0]
        )
        bottom_right = self.params["W_2"] @ self.params["W_2"].T + self.c[1] * jnp.eye(
            self.n_features_[1]
        )
        top_right = self.params["W_1"] @ self.params["W_2"].T
        bottom_left = self.params["W_2"] @ self.params["W_1"].T

        # Construct the matrix using the blocks
        matrix = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        return matrix
