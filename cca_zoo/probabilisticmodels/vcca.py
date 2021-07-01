import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

# import arviz as az
from cca_zoo.models import _CCA_Base


class VariationalCCA(_CCA_Base):
    """
    A class used to fit a variational bayesian CCA

    Citation
    --------
    Wang, Chong. "Variational Bayesian approach to canonical correlation analysis." IEEE Transactions on Neural Networks 18.3 (2007): 905-910.

    :Example:

    >>> from cca_zoo.probabilisticmodels import VariationalCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = VariationalCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state: int = 0):
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, accept_sparse=True,
                         random_state=random_state)

    def fit(self, *views: np.ndarray):
        nuts_kernel = NUTS(self.model)
        self.mcmc = MCMC(nuts_kernel, num_samples=100, num_warmup=100)
        rng_key = random.PRNGKey(0)
        self.mcmc.run(rng_key, *views)
        self.posterior_samples = self.mcmc.get_samples()
        return self

    def model(self, *views: np.ndarray):
        n = views[0].shape[0]
        p = [view.shape[1] for view in views]
        # mean of column in each view of data (p_1,)
        mu = [numpyro.sample("mu_" + str(i), dist.MultivariateNormal(0., 10 * jnp.eye(p_))) for i, p_ in enumerate(p)]
        """
        Generates cholesky factors of correlation matrices using an LKJ prior.

        The expected use is to combine it with a vector of variances and pass it
        to the scale_tril parameter of a multivariate distribution such as MultivariateNormal.

        E.g., if theta is a (positive) vector of covariances with the same dimensionality
        as this distribution, and Omega is sampled from this distribution,
        scale_tril=torch.mm(torch.diag(sqrt(theta)), Omega)
        """
        psi = [numpyro.sample("psi_" + str(i), dist.LKJCholesky(p_)) for i, p_ in enumerate(p)]
        # sample weights to get from latent to data space (k,p)
        with numpyro.plate("plate_views", self.latent_dims):
            self.weights_list = [numpyro.sample("W_" + str(i), dist.MultivariateNormal(0., jnp.diag(jnp.ones(p_)))) for
                                 i, p_ in enumerate(p)]
        with numpyro.plate("plate_i", n):
            # sample from latent z - normally disributed (n,k)
            z = numpyro.sample("z", dist.MultivariateNormal(0., jnp.diag(jnp.ones(self.latent_dims))))
            # sample from multivariate normal and observe data
            [numpyro.sample("obs" + str(i), dist.MultivariateNormal((z @ W_) + mu_, scale_tril=psi_), obs=X_) for
             i, (X_, psi_, mu_, W_) in enumerate(zip(views, psi, mu, self.weights_list))]


def main():
    np.random.seed(42)
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)
    vcca = VariationalCCA(latent_dims=1).fit(X, Y)
    print()


if __name__ == "__main__":
    main()
