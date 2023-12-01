import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from cca_zoo.probabilistic._cca import ProbabilisticCCA


class ProbabilisticRCCA(ProbabilisticCCA):
    """
    Probabilistic Ridge Canonical Correlation Analysis (Probabilistic Ridge CCA).

    Probabilistic Ridge CCA extends the Probabilistic Canonical Correlation Analysis model
    by introducing regularization terms in the linear relationships between multiple representations
    of data. This regularization improves the conditioning of the problem and provides a
    way to incorporate prior knowledge. It combines features of both CCA and Ridge Regression.

    Parameters
    ----------

    c: float, default=1.0
        Regularization strength; must be a positive float. Regularization improves
        the conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization.

    References
    ----------

    [1] De Bie, T. and De Moor, B., 2003. On the regularization of canonical correlation analysis. Int. Sympos. ICA and BSS, pp.785-790.
    """

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

        sigma1 = numpyro.param(
            "sigma_1", jnp.ones(1), constraint=dist.constraints.positive
        )
        sigma2 = numpyro.param(
            "sigma_2", jnp.ones(1), constraint=dist.constraints.positive
        )
        psi1 = jnp.eye(self.n_features_in_[0]) * sigma1
        psi2 = jnp.eye(self.n_features_in_[1]) * sigma2

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

    def joint(self):
        psi1 = jnp.eye(self.n_features_in_[0]) * self.params["sigma_1"]
        psi2 = jnp.eye(self.n_features_in_[1]) * self.params["sigma_2"]
        # Calculate the individual matrix blocks
        top_left = self.params["W_1"] @ self.params["W_1"].T + psi1
        bottom_right = self.params["W_2"] @ self.params["W_2"].T + psi2
        top_right = self.params["W_1"] @ self.params["W_2"].T
        bottom_left = self.params["W_2"] @ self.params["W_1"].T

        # Construct the matrix using the blocks
        matrix = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        return matrix
