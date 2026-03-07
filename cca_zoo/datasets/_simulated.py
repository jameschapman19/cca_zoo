"""Simulated multiview data generator via a linear latent variable model."""

from __future__ import annotations

import numpy as np


class JointData:
    """Generate multiview data from a linear latent variable model.

    Each view is generated as::

        x_i = Z @ W_i.T + noise_i

    where Z ~ N(0, I_{k x k}) is the shared latent variable,
    W_i ~ N(0, I) is the view-specific loading matrix, and
    noise_i ~ N(0, sigma_i^2 I) is independent noise with variance
    controlled by ``signal_to_noise``.

    Args:
        n_views: Number of views to generate. Default is 2.
        n_samples: Number of observations to generate. Default is 100.
        latent_dimensions: Dimension of the shared latent space. Default
            is 1.
        n_features: Number of features per view.  May be a single integer
            (same for all views) or a list of integers with one entry per
            view. Default is 10.
        signal_to_noise: Signal-to-noise ratio.  May be a single float
            (same for all views) or a list of floats. A higher value
            means less noise.  Default is 1.0.
        random_state: Integer seed or ``None`` for reproducibility.

    Example:
        >>> import numpy as np
        >>> data = JointData(
        ...     n_views=2, n_samples=100, latent_dimensions=2, random_state=0
        ... )
        >>> views = data.sample()
        >>> len(views)
        2
        >>> views[0].shape
        (100, 10)
    """

    def __init__(
        self,
        n_views: int = 2,
        n_samples: int = 100,
        latent_dimensions: int = 1,
        n_features: int | list[int] = 10,
        signal_to_noise: float | list[float] = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_views = n_views
        self.n_samples = n_samples
        self.latent_dimensions = latent_dimensions
        self.n_features = n_features
        self.signal_to_noise = signal_to_noise
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._features_per_view: list[int] = self._broadcast_param(
            n_features, n_views, "n_features"
        )
        self._snr_per_view: list[float] = self._broadcast_param(
            signal_to_noise, n_views, "signal_to_noise"
        )

        # Pre-generate weight matrices so successive calls to sample()
        # share the same generative parameters.
        self._weights: list[np.ndarray] = [
            self._rng.standard_normal((p, latent_dimensions))
            for p in self._features_per_view
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _broadcast_param(
        value: int | float | list[int] | list[float],
        n_views: int,
        name: str,
    ) -> list[float]:
        """Broadcast a scalar or list parameter to a list of length n_views.

        Args:
            value: Scalar or list to broadcast.
            n_views: Required list length.
            name: Parameter name for error messages.

        Returns:
            List of floats with exactly ``n_views`` elements.

        Raises:
            ValueError: If ``value`` is a list with wrong length.
        """
        if isinstance(value, (int, float)):
            return [float(value)] * n_views
        items = list(value)
        if len(items) != n_views:
            raise ValueError(
                f"Parameter '{name}' must be a scalar or a list of length "
                f"{n_views}, got {len(items)}."
            )
        return [float(v) for v in items]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self) -> list[np.ndarray]:
        """Draw a new set of multiview samples from the generative model.

        Returns:
            List of numpy arrays, one per view, each of shape
            (n_samples, n_features_i).
        """
        z = self._rng.standard_normal((self.n_samples, self.latent_dimensions))
        views: list[np.ndarray] = []
        for w, snr in zip(self._weights, self._snr_per_view):
            signal = z @ w.T  # (n_samples, p_i)
            noise_std = 1.0 / np.sqrt(snr) if snr > 0 else 1.0
            noise = self._rng.standard_normal(signal.shape) * noise_std
            views.append(signal + noise)
        return views

    def __call__(self) -> list[np.ndarray]:
        """Alias for :meth:`sample`.

        Args:
            None

        Returns:
            List of numpy arrays, one per view.

        Example:
            >>> data = JointData(n_views=2, n_samples=50, random_state=1)
            >>> views = data()
            >>> len(views)
            2
        """
        return self.sample()
