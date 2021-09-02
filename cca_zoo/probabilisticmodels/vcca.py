import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models import _CCA_Base


class VariationalCCA(_CCA_Base):
    """
    A class used to fit a variational bayesian CCA

    Citation
    --------
    Wang, Chong. "Variational Bayesian approach to canonical correlation analysis." IEEE Transactions on Neural Networks 18.3 (2007): 905-910.

    :Example:


    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state: int = 0,
                 num_samples=100, num_warmup=100):
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, accept_sparse=True,
                         random_state=random_state)
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.rng_key = PRNGKey(random_state)

    def fit(self, *views: np.ndarray):
        """
        Infer the parameters (mu: mean, psi: within view variance) and latent variables (z) of the generative CCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        nuts_kernel = NUTS(self.model)
        self.mcmc = MCMC(nuts_kernel, num_samples=self.num_samples, num_warmup=self.num_warmup)
        self.mcmc.run(self.rng_key, *views)
        self.posterior_samples = self.mcmc.get_samples()
        return self

    def transform(self, *views):
        """
        Predict the latent variables that generate the data in views using the sampled model parameters

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        check_is_fitted(self, attributes=['posterior_samples'])
        return Predictive(self.model, self.posterior_samples, return_sites=['z'])(
            PRNGKey(1), *views)['z']

    def model(self, *views: np.ndarray):
        n = views[0].shape[0]
        p = [view.shape[1] for view in views]
        # parameter representing the mean of column in each view of data
        mu = [numpyro.sample("mu_" + str(i), dist.MultivariateNormal(0., 10 * jnp.eye(p_))) for i, p_ in enumerate(p)]
        # parameter representing the within view variance for each view of data
        psi = [numpyro.sample("psi_" + str(i), dist.LKJCholesky(p_)) for i, p_ in enumerate(p)]
        # parameter representing weights applied to latent variables
        with numpyro.plate("plate_views", self.latent_dims):
            self.weights_list = [numpyro.sample("W_" + str(i), dist.MultivariateNormal(0., jnp.diag(jnp.ones(p_)))) for
                                 i, p_ in enumerate(p)]
        with numpyro.plate("plate_i", n):
            # sample from latent z: the latent variables of the model
            z = numpyro.sample("z", dist.MultivariateNormal(0., jnp.diag(jnp.ones(self.latent_dims))))
            # sample from multivariate normal and observe data
            [numpyro.sample("obs" + str(i), dist.MultivariateNormal((z @ W_) + mu_, scale_tril=psi_), obs=X_) for
             i, (X_, psi_, mu_, W_) in enumerate(zip(views, psi, mu, self.weights_list))]
