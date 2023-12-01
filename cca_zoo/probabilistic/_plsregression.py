from typing import Iterable

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

from cca_zoo.probabilistic._cca import ProbabilisticCCA


class ProbabilisticPLSRegression(ProbabilisticCCA):
    """
    Probabilistic Ridge Canonical Correlation Analysis (Probabilistic Ridge CCA).

    Probabilistic Ridge CCA extends the Probabilistic Canonical Correlation Analysis model
    by introducing regularization terms in the linear relationships between multiple representations
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

    def _model(self, views):
        """
        Defines the generative model for Probabilistic RCCA.

        Parameters
        ----------
        views: tuple of np.ndarray
            A tuple containing the first and second representations, X1 and X2, each as a numpy array.
        """
        X1, X2 = views

        W = numpyro.param(
            "W",
            random.normal(
                shape=(
                    self.n_features_in_[0],
                    self.latent_dimensions,
                ),
                key=self.rng_key,
            ),
        )
        C = numpyro.param(
            "C",
            random.normal(
                shape=(
                    self.n_features_in_[1],
                    self.latent_dimensions,
                ),
                key=self.rng_key,
            ),
        )

        B = numpyro.param(
            "B",
            jnp.ones(
                shape=(self.latent_dimensions,),
            ),
        )

        # Add positive-definite constraint for psi1 and psi2
        e = numpyro.param(
            "e",
            jnp.ones(shape=(self.n_features_in_[0],)),
        )
        f = numpyro.param(
            "f",
            jnp.ones(shape=(self.n_features_in_[1],)),
        )
        h = numpyro.param(
            "h",
            jnp.ones(shape=(self.latent_dimensions,)),
            constraint=dist.constraints.positive,
        )

        n_samples = X1.shape[0] if X1 is not None else X2.shape[0]

        with numpyro.plate("n", n_samples):
            t = numpyro.sample(
                "t",
                dist.MultivariateNormal(
                    jnp.zeros(self.latent_dimensions), jnp.eye(self.latent_dimensions)
                ),
            )
            u = numpyro.sample(
                "u",
                dist.MultivariateNormal(t * B, jnp.diag(h)),
            )

        with numpyro.plate("n", n_samples):
            numpyro.sample(
                "X1",
                dist.MultivariateNormal(t @ W.T, covariance_matrix=jnp.diag(e)),
                obs=X1,
            )
            numpyro.sample(
                "X2",
                dist.MultivariateNormal(u @ C.T, covariance_matrix=jnp.diag(f)),
                obs=X2,
            )

    def _guide(self, views):
        """
        Defines the variational family (guide) for approximate inference in Probabilistic CCA.

        Parameters
        ----------
        views: tuple of np.ndarray
            A tuple containing the first and second representations, X1 and X2, each as a numpy array.
        """
        X1, X2 = views

        n = X1.shape[0] if X1 is not None else X2.shape[0]

        with numpyro.plate("n", n):
            numpyro.sample(
                "t",
                dist.MultivariateNormal(
                    jnp.zeros(self.latent_dimensions), jnp.eye(self.latent_dimensions)
                ),
            )
            numpyro.sample(
                "u",
                dist.MultivariateNormal(
                    jnp.zeros(self.latent_dimensions), jnp.eye(self.latent_dimensions)
                ),
            )

    def transform(self, views: Iterable[np.ndarray], y=None, return_std=False):
        """
        Transform the data into the latent space.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            A list or tuple of numpy arrays representing different representations of the same samples. Each numpy array must have the same number of rows.
        y: Any, optional
            Ignored in this implementation.

        Returns
        -------
        representations : np.ndarray
            The transformed data in the latent space.
        """
        conditioned_model = handlers.substitute(self._model, self.params)
        kernel = NUTS(conditioned_model)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)
        mcmc.run(self.rng_key, views)
        samples = mcmc.get_samples()
        t = samples["t"]
        if return_std:
            return np.array(t.mean(axis=0)), np.array(t.std(axis=0))
        else:
            return np.array(t.mean(axis=0))

    def joint(self):
        # Calculate the individual matrix blocks
        top_left = self.params["W"] @ self.params["W"].T + jnp.diag(self.params["e"])
        top_right = self.params["W"] @ jnp.diag(self.params["B"]) @ self.params["C"].T
        bottom_left = self.params["C"] @ jnp.diag(self.params["B"]) @ self.params["W"].T
        bottom_right = self.params["C"] @ (
            jnp.diag(self.params["B"] ** 2) + jnp.diag(self.params["h"])
        ) @ self.params["C"].T + jnp.diag(self.params["f"])

        # Construct the matrix using the blocks
        matrix = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        return matrix


if __name__ == "__main__":
    t = np.random.normal(size=(100, 1))
    b = np.random.normal(size=(1,))
    u = t * b + np.random.normal(size=(100, 1)) / 10
    w = np.random.normal(size=(1, 5))
    c = np.random.normal(size=(1, 5))
    X = w * t + np.random.normal(size=(100, 5)) / 10
    Y = c * u + np.random.normal(size=(100, 5)) / 10
    from cca_zoo.linear import CCA, PLS

    # Models and fit
    cca = CCA(latent_dimensions=1)
    pls = PLS(latent_dimensions=1)
    ppls = ProbabilisticPLSRegression(latent_dimensions=1, random_state=1, n_iter=50000)

    cca.fit([X, Y])
    pls.fit([X, Y])
    ppls.fit([X, Y])
    model_joint = ppls.joint()

    # Assert: Calculate correlation coefficient and ensure it's greater than 0.98
    z_cca = cca.transform([X, Y])[0]
    z_pls = pls.transform([X, Y])[0]
    z_p, z_pstd = np.array(ppls.transform([X, None], return_std=True))
    # correlation between pls and ppls
    correlation_matrix = np.abs(np.corrcoef(z_pls.reshape(-1), z_p.reshape(-1)))
    correlation_pls = correlation_matrix[0, 1]

    correlation_matrix = np.abs(np.corrcoef(z_cca.reshape(-1), z_p.reshape(-1)))
    correlation_cca = correlation_matrix[0, 1]

    S = np.cov(X.T, Y.T)

    assert (
        correlation_pls > correlation_cca
    ), f"Expected correlation with PLS greater than CCA, got {correlation_pls} and {correlation_cca}"
