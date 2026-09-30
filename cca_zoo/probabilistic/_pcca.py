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
    maximum_likelihood_start,
    pcca_model,
)


class ProbabilisticCCA(BaseProbabilistic):
    r"""Probabilistic CCA with posterior sampling by NUTS.

    $$
    z \sim \mathcal{N}(0, I), \qquad
    x_i \mid z \sim \mathcal{N}(W_i z + \mu_i, \Psi_i),
    $$

    with each $\Psi_i$ a full covariance, so that $z$ models only what the
    views share: at the maximum likelihood each view's posterior mean spans
    its canonical variates (Bach & Jordan, 2005). $\Psi_i$ has an LKJ prior
    on its correlation and log-normal scales, and $z$ is integrated out. The
    likelihood is invariant to a shared rotation of the loadings, so the
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

    def _model(self, views: list[np.ndarray]) -> None:
        """Numpyro model on centred views, unit-variance prior on the loadings."""
        pcca_model(views, self.n_components)

    def fit(self, views: list[ArrayLike], y: None = None) -> ProbabilisticCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        import jax
        from numpyro.infer import MCMC, NUTS, init_to_value

        validated = self._setup_fit(views)

        # Sampling starts at the maximum likelihood.
        start = maximum_likelihood_start(validated, self.n_components)
        nuts_kernel = NUTS(self._model, init_strategy=init_to_value(values=start))
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
        # before averaging: align every view's stacked W_i draws to a common
        # reference.
        w_stack = np.concatenate(
            [self.posterior_samples_[f"W_{i}"] for i in range(self.n_views_)], axis=1
        )  # (n_posterior_samples, P, k)
        aligned_w, _ = align_posterior_rotation(w_stack)
        splits = np.cumsum(self.n_features_per_view_)[:-1]
        for i, w_i_aligned in enumerate(np.split(aligned_w, splits, axis=1)):
            self.posterior_samples_[f"W_{i}"] = w_i_aligned

        # Set weights_ to posterior mean W matrices (p_i x k) for each view
        self.weights_: list[np.ndarray] = [
            self.posterior_samples_[f"W_{i}"].mean(axis=0) for i in range(self.n_views_)
        ]
        self._fit_maps_and_importances(validated)
        return self
