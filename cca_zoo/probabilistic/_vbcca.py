"""VariationalBayesCCA — Bayesian CCA with ARD via variational inference (numpyro)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo.probabilistic._utils import PosteriorMeanTransformMixin

# Weak, near-uninformative Gamma hyperprior on each ARD precision alpha_k,
# following the standard choice for automatic relevance determination in
# Bayesian PCA/CCA (e.g. Bishop 1999; Wang 2007).
_ARD_A0 = 1e-3
_ARD_B0 = 1e-3


class VariationalBayesCCA(PosteriorMeanTransformMixin, BaseModel):
    r"""Variational Bayesian CCA with automatic relevance determination.

    Fits the same probabilistic CCA generative model as
    :class:`~cca_zoo.probabilistic.ProbabilisticCCA`, extended with a
    hierarchical automatic relevance determination (ARD) prior over the
    columns of the loading matrices, shared across views:

    $$
    \begin{aligned}
    \alpha_k &\sim \mathrm{Gamma}(a_0, b_0), & k &= 1, \dots, K \\
    W_i[:, k] &\sim \mathcal{N}(0,\ \alpha_k^{-1} I), & i &= 1, \dots, V \\
    z &\sim \mathcal{N}(0, I_K) \\
    x_i \mid z &\sim \mathcal{N}(W_i z + \mu_i,\ \Psi_i)
    \end{aligned}
    $$

    Because $\alpha_k$ is shared across every view's $k$-th loading column,
    a latent dimension is only retained if *some* view finds it useful;
    dimensions unsupported by the data are shrunk towards zero in every view
    simultaneously. The posterior mean of $\alpha_k$ (exposed as
    ``ard_relevance_``) is therefore a direct, per-dimension usefulness
    score: large values indicate a dimension that has been shrunk away and
    can be dropped, giving automatic latent-dimensionality selection instead
    of a `GridSearchCV` sweep over `latent_dimensions`.

    Inference uses mean-field stochastic variational inference (SVI) via
    numpyro, rather than the closed-form conjugate coordinate-ascent updates
    derived in Wang (2007) for this model: SVI reuses the exact same
    ``numpyro`` generative-model machinery as
    :class:`~cca_zoo.probabilistic.ProbabilisticCCA`, and (unlike a
    hand-derived conjugate solver) extends unmodified to non-conjugate
    variants of the model. It is a substantially cheaper alternative to that
    class's full NUTS MCMC, at the cost of the mean-field independence
    assumption between latent variables.

    The ``weights_`` attribute is set to the variational posterior mean of
    each $W_i$ matrix so that :class:`~cca_zoo._base.BaseModel`'s scoring
    utilities work without modification.

    References:
        Bach, F. R. & Jordan, M. I. "A probabilistic interpretation of
        canonical correlation analysis." (2005).
        Wang, C. "Variational Bayesian approach to canonical correlation
        analysis." IEEE Transactions on Neural Networks 18.3 (2007).

    Args:
        latent_dimensions: Dimensionality of the latent space. Default is 1.
            Because of the ARD prior, this should be set generously (an
            upper bound on the number of shared factors you expect); use
            ``ard_relevance_`` after fitting to see how many were retained.
        center: Whether to center each view before fitting. Default is True.
        num_steps: Number of SVI gradient steps. Default is 2000.
        learning_rate: Adam learning rate for SVI. Default is 1e-2.
        num_posterior_samples: Number of samples drawn from the fitted
            variational posterior to populate ``posterior_samples_``.
            Default is 1000.
        random_state: Integer seed for JAX PRNG. Default is 0.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 4))
        >>> X2 = rng.standard_normal((50, 3))
        >>> model = VariationalBayesCCA(
        ...     latent_dimensions=2, num_steps=50
        ... ).fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        num_steps: int = 2000,
        learning_rate: float = 1e-2,
        num_posterior_samples: int = 1000,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.num_posterior_samples = num_posterior_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    # numpyro generative model
    # ------------------------------------------------------------------

    def _model(self, views: list[np.ndarray]) -> None:
        """Numpyro generative model for ARD variational Bayesian CCA.

        Args:
            views: List of centered arrays, each (n_samples, n_features_i).
        """
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        n = views[0].shape[0]
        k = self.latent_dimensions

        # Shared ARD precision per latent dimension, tying all views' loading
        # columns together so shrinkage decisions are made jointly.
        alpha = numpyro.sample(
            "alpha",
            dist.Gamma(jnp.full((k,), _ARD_A0), jnp.full((k,), _ARD_B0)).to_event(1),
        )
        scale = 1.0 / jnp.sqrt(alpha)  # (k,)

        # Sample per-view parameters
        ws: list[Any] = []
        psis: list[Any] = []
        for i, xi in enumerate(views):
            p_i = xi.shape[1]
            w_i = numpyro.sample(
                f"W_{i}",
                dist.Normal(
                    jnp.zeros((p_i, k)), jnp.broadcast_to(scale, (p_i, k))
                ).to_event(2),
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

    def fit(self, views: list[ArrayLike], y: None = None) -> VariationalBayesCCA:
        """Run mean-field SVI to infer an approximate posterior.

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
        import numpyro.optim as optim
        from numpyro.infer import SVI, Predictive, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        validated = self._setup_fit(views)

        guide = AutoNormal(self._model)
        svi = SVI(self._model, guide, optim.Adam(self.learning_rate), Trace_ELBO())

        rng_key, predictive_key = jax.random.split(
            jax.random.PRNGKey(self.random_state)
        )
        svi_result = svi.run(rng_key, self.num_steps, validated, progress_bar=False)
        self.svi_result_ = svi_result
        self.losses_: np.ndarray = np.array(svi_result.losses)
        self.guide_ = guide

        predictive = Predictive(
            guide, params=svi_result.params, num_samples=self.num_posterior_samples
        )
        self.posterior_samples_: dict[str, Any] = predictive(predictive_key, validated)

        # Set weights_ to variational posterior mean W matrices (p_i x k)
        self.weights_: list[np.ndarray] = [
            np.array(self.posterior_samples_[f"W_{i}"].mean(axis=0))
            for i in range(self.n_views_)
        ]
        # Posterior mean ARD precision per latent dimension: larger means
        # "more shrunk / less relevant".
        self.ard_relevance_: np.ndarray = np.array(
            self.posterior_samples_["alpha"].mean(axis=0)
        )
        return self
