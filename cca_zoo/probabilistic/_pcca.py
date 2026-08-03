"""ProbabilisticCCA — Bayesian CCA via NUTS MCMC (numpyro)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo.probabilistic._utils import PosteriorMeanTransformMixin


class ProbabilisticCCA(PosteriorMeanTransformMixin, BaseModel):
    r"""Probabilistic Canonical Correlation Analysis via NUTS MCMC.

    Fits a Bayesian latent variable model with the following generative
    process for $V$ views:

    $$
    \begin{aligned}
    z &\sim \mathcal{N}(0, I) \\
    x_i \mid z &\sim \mathcal{N}(W_i z + \mu_i,\ \Psi_i), \quad i = 1, \dots, V
    \end{aligned}
    $$

    MCMC sampling is performed with the No-U-Turn Sampler (NUTS) from
    numpyro.  After fitting, :meth:`transform` returns the posterior
    mean of z conditioned on the observed views (computed analytically
    using the posterior mean formula for linear Gaussian models).

    The ``weights_`` attribute is set to the posterior mean of each W_i
    matrix so that :class:`~cca_zoo._base.BaseModel`'s scoring utilities
    work without modification.

    References:
        Bach, F. R. & Jordan, M. I. "A probabilistic interpretation of
        canonical correlation analysis." (2005).
        Wang, C. "Variational Bayesian approach to canonical correlation
        analysis." IEEE Transactions on Neural Networks 18.3 (2007).

    Args:
        latent_dimensions: Dimensionality of the latent space. Default is 1.
        center: Whether to center each view before fitting. Default is True.
        num_warmup: Number of NUTS warm-up (burn-in) steps. Default is 500.
        num_samples: Number of NUTS posterior samples to draw. Default is 1000.
        random_state: Integer seed for JAX PRNG. Default is 0.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 4))
        >>> X2 = rng.standard_normal((50, 3))
        >>> model = ProbabilisticCCA(
        ...     latent_dimensions=2, num_warmup=10, num_samples=10
        ... ).fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        num_warmup: int = 500,
        num_samples: int = 1000,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    # numpyro generative model
    # ------------------------------------------------------------------

    def _model(self, views: list[np.ndarray]) -> None:
        """Numpyro generative model for probabilistic CCA.

        Args:
            views: List of centered arrays, each (n_samples, n_features_i).
        """
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        n = views[0].shape[0]
        k = self.latent_dimensions

        # Sample per-view parameters
        ws: list[Any] = []
        psis: list[Any] = []
        for i, xi in enumerate(views):
            p_i = xi.shape[1]
            w_i = numpyro.sample(
                f"W_{i}",
                dist.Normal(jnp.zeros((p_i, k)), jnp.ones((p_i, k))).to_event(2),
            )
            log_psi_i = numpyro.sample(
                f"log_psi_{i}",
                dist.Normal(jnp.zeros(p_i), jnp.ones(p_i)).to_event(1),
            )
            ws.append(w_i)
            psis.append(jnp.exp(log_psi_i))

        # Sample latent variables and observations
        with numpyro.plate("n", n):
            z = numpyro.sample(
                "z",
                dist.Normal(jnp.zeros(k), jnp.ones(k)).to_event(1),
            )
            for i, (xi, w_i, psi_i) in enumerate(zip(views, ws, psis)):
                mean_i = z @ w_i.T  # (n, p_i)
                numpyro.sample(
                    f"x_{i}",
                    dist.Normal(mean_i, psi_i).to_event(1),
                    obs=jnp.array(xi),
                )

    # ------------------------------------------------------------------
    # Public fit / transform
    # ------------------------------------------------------------------

    def fit(self, views: list[ArrayLike], y: None = None) -> ProbabilisticCCA:
        """Run NUTS MCMC to infer posterior over model parameters and latents.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
                All arrays must have the same number of rows.
            y: Ignored.  Present for scikit-learn API compatibility.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        import jax
        from numpyro.infer import MCMC, NUTS

        validated = self._setup_fit(views)

        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
        )
        rng_key = jax.random.PRNGKey(self.random_state)
        mcmc.run(rng_key, validated)
        self.mcmc_ = mcmc
        self.posterior_samples_: dict[str, Any] = mcmc.get_samples()

        # Set weights_ to posterior mean W matrices (p_i x k) for each view
        self.weights_: list[np.ndarray] = [
            np.array(self.posterior_samples_[f"W_{i}"].mean(axis=0))
            for i in range(self.n_views_)
        ]
        return self
