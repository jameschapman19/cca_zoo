"""MCCA_EY — Multiview Eckart-Young CCA."""

from __future__ import annotations

from numpy.typing import ArrayLike

from cca_zoo.linear.gradient._cca_ey import CCA_EY


class MCCA_EY(CCA_EY):
    r"""Eckart-Young multiview CCA for large-scale data (>=2 views).

    Identical to :class:`CCA_EY`; the shared Eckart-Young loss and its
    gradient (see :mod:`cca_zoo._utils._ey`) are already defined for an
    arbitrary number of views, so no multiview-specific logic is needed here.

    References:
        Chapman, J., Lawry Aguila, A., & Wells, L. (2022). A Generalised
        EigenGame with Extensions to Multiview Representation Learning.
        arXiv:2211.11323.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        c: Ridge regularisation parameter(s) in ``[0, 1]``.  Default is 0.
        learning_rate: Gradient step size. Default is 1e-2.
        max_iter: Number of gradient steps. Default is 1000.
        batch_size: Mini-batch size.  ``None`` uses the full dataset.
        tol: Convergence tolerance. Default is 1e-6.
        momentum: Momentum coefficient in ``[0, 1)``. Default is 0.9.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 500))
        >>> X2 = rng.standard_normal((200, 400))
        >>> X3 = rng.standard_normal((200, 300))
        >>> model = MCCA_EY(latent_dimensions=4, c=0.1, batch_size=64, random_state=0)
        >>> model = model.fit([X1, X2, X3])
    """

    def fit(self, views: list[ArrayLike], y: None = None) -> MCCA_EY:
        """Fit MCCA_EY for 2 or more views.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        super().fit(views, y)
        return self
