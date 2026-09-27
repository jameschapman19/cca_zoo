"""Variational Bayesian CCA with automatic relevance determination."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._param_constraints import POSITIVE_EPS, POSITIVE_INT, RANDOM_STATE
from cca_zoo.probabilistic._utils import BaseProbabilistic, _integer_seed

# Weak, near-uninformative Gamma hyperprior on each ARD precision alpha_k,
# following the standard choice for automatic relevance determination in
# Bayesian PCA/CCA (e.g. Bishop 1999; Wang 2007).
_ARD_A0 = 1e-3
_ARD_B0 = 1e-3


class VariationalBayesCCA(BaseProbabilistic):
    r"""Variational Bayesian CCA with automatic relevance determination.

    The probabilistic CCA model with an ARD prior shared across views:

    $$
    \begin{aligned}
    \alpha_k &\sim \mathrm{Gamma}(a_0, b_0), &
    W_i[:, k] &\sim \mathcal{N}(0, \alpha_k^{-1} I), \\
    z &\sim \mathcal{N}(0, I), &
    x_i \mid z &\sim \mathcal{N}(W_i z + \mu_i, \Psi_i).
    \end{aligned}
    $$

    Dimensions no view supports are shrunk away, so set ``n_components``
    generously and read ``ard_relevance_``. Inference is mean-field SVI in
    numpyro, cheaper than :class:`ProbabilisticCCA`'s NUTS. Requires the
    ``probabilistic`` extra.

    Args:
        n_components: Upper bound on the number of latent dimensions.
            Default is 1.
        center: Whether to centre each view. Default is True.
        n_iter: SVI steps; there is no stopping rule, so all are run.
            Default is 2000.
        learning_rate: Adam learning rate for SVI. Default is 1e-2.
        n_posterior_samples: Draws from the fitted posterior. Default is 1000.
        random_state: Seed for the JAX PRNG. Default is None.

    Attributes:
        weights_: Posterior mean loadings of each view, shape
            (n_features_i, n_components).
        ard_relevance_: Posterior mean ARD precision of each dimension;
            large means shrunk away.
        posterior_samples_: Posterior draws keyed by site name.
        losses_: SVI loss at each step.

    References:
        Wang, C. (2007). Variational Bayesian approach to canonical
        correlation analysis. IEEE Transactions on Neural Networks, 18(3),
        905-910.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.probabilistic import VariationalBayesCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 4))
        >>> X2 = rng.standard_normal((50, 3))
        >>> model = VariationalBayesCCA(n_components=2, n_iter=50).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseProbabilistic._parameter_constraints,
        "n_iter": POSITIVE_INT,
        "learning_rate": POSITIVE_EPS,
        "n_posterior_samples": POSITIVE_INT,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        n_iter: int = 2000,
        learning_rate: float = 1e-2,
        n_posterior_samples: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.n_posterior_samples = n_posterior_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    # numpyro generative model
    # ------------------------------------------------------------------

    def _model(self, views: list[np.ndarray]) -> None:
        """Numpyro generative model on centred views."""
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        n = views[0].shape[0]
        k = self.n_components

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
            psis.append(jnp.exp(log_psi_i))  # noise variances

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
                    dist.Normal(mean_i, jnp.sqrt(psi_i)).to_event(1),
                    obs=jnp.array(xi),
                )

    # ------------------------------------------------------------------
    # Public fit / transform
    # ------------------------------------------------------------------

    def fit(self, views: list[ArrayLike], y: None = None) -> VariationalBayesCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        import jax
        import numpyro.optim as optim
        from numpyro.infer import SVI, Predictive, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        validated = self._setup_fit(views)

        guide = AutoNormal(self._model)
        svi = SVI(self._model, guide, optim.Adam(self.learning_rate), Trace_ELBO())

        rng_key, predictive_key = jax.random.split(
            jax.random.PRNGKey(_integer_seed(self.random_state))
        )
        svi_result = svi.run(rng_key, self.n_iter, validated, progress_bar=False)
        self.losses_: np.ndarray = np.array(svi_result.losses)

        predictive = Predictive(
            guide, params=svi_result.params, num_samples=self.n_posterior_samples
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
        return self._finish_fit(validated)
