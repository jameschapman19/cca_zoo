"""Probabilistic CCA by NUTS."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._param_constraints import POSITIVE_INT, RANDOM_STATE
from cca_zoo.probabilistic._utils import (
    BaseProbabilistic,
    _integer_seed,
    align_posterior_rotation,
)


class ProbabilisticCCA(BaseProbabilistic):
    r"""Probabilistic CCA with posterior sampling by NUTS.

    $$
    z \sim \mathcal{N}(0, I), \qquad
    x_i \mid z \sim \mathcal{N}(W_i z + \mu_i, \Psi_i).
    $$

    The likelihood is invariant to a shared rotation of the loadings, so the
    draws are aligned by generalized Procrustes before averaging. Requires
    the ``probabilistic`` extra.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        n_warmup: NUTS warm-up steps. Default is 500.
        n_posterior_samples: NUTS draws. Default is 1000.
        random_state: Seed for the JAX PRNG. Default is None.

    Attributes:
        weights_: Posterior mean loadings of each view, shape
            (n_features_i, n_components).
        posterior_samples_: Aligned posterior draws keyed by site name.

    References:
        Bach, F. R., & Jordan, M. I. (2005). A probabilistic interpretation
        of canonical correlation analysis. Technical Report 688, UC Berkeley.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.probabilistic import ProbabilisticCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 4))
        >>> X2 = rng.standard_normal((50, 3))
        >>> model = ProbabilisticCCA(
        ...     n_components=2, n_warmup=10, n_posterior_samples=10
        ... ).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseProbabilistic._parameter_constraints,
        "n_warmup": POSITIVE_INT,
        "n_posterior_samples": POSITIVE_INT,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        n_warmup: int = 500,
        n_posterior_samples: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.n_warmup = n_warmup
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

    def fit(self, views: list[ArrayLike], y: None = None) -> ProbabilisticCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        import jax
        from numpyro.infer import MCMC, NUTS

        validated = self._setup_fit(views)

        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=self.n_warmup,
            num_samples=self.n_posterior_samples,
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(_integer_seed(self.random_state))
        mcmc.run(rng_key, validated)
        self.posterior_samples_ = {
            k: np.array(v) for k, v in mcmc.get_samples().items()
        }

        # Resolve the model's rotational symmetry (see class docstring)
        # before any cross-draw averaging: stack every view's W_i draws,
        # align them to a common reference, then rotate that draw's z by
        # the same rotation to keep it internally consistent.
        w_stack = np.concatenate(
            [self.posterior_samples_[f"W_{i}"] for i in range(self.n_views_)], axis=1
        )  # (n_posterior_samples, P, k)
        aligned_w, rotations = align_posterior_rotation(w_stack)
        splits = np.cumsum(self.n_features_per_view_)[:-1]
        for i, w_i_aligned in enumerate(np.split(aligned_w, splits, axis=1)):
            self.posterior_samples_[f"W_{i}"] = w_i_aligned
        self.posterior_samples_["z"] = np.einsum(
            "snk,skj->snj", self.posterior_samples_["z"], rotations
        )

        # Set weights_ to posterior mean W matrices (p_i x k) for each view
        self.weights_: list[np.ndarray] = [
            self.posterior_samples_[f"W_{i}"].mean(axis=0) for i in range(self.n_views_)
        ]
        self._fit_maps_and_importances(validated)
        return self
