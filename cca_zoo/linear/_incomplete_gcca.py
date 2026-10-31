"""Generalized CCA for samples with missing views."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import psd_inverse_sqrt
from cca_zoo._utils._param_constraints import RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter


class IncompleteGCCA(BaseModel):
    r"""Generalized CCA when some samples lack some views.

    A sample's view is missing when its whole row is ``NaN``, as when an assay
    was not run on that sample. Each sample's latent score $g_j$ is fitted to
    the views it has:

    $$
    \min_{G,\, w_i} \sum_i \Bigl( (n_i - 1)\, w_i^\top C_i w_i
        - 2\, w_i^\top X_i^\top G_i + \operatorname{tr}(G_i^\top G_i) \Bigr)
    \quad \text{subject to} \quad G^\top D G = s I,
    $$

    where $X_i$ and $G_i$ hold the $n_i$ samples with view $i$, $D$ is the
    diagonal matrix of each sample's number of observed views, $s = (n - 1)
    \bar d$ for $\bar d$ the mean of $D$'s diagonal, and $C_i = (1 - c_i)
    \Sigma_i + c_i I$ with $\Sigma_i$ the covariance of the $n_i$ observed
    rows. Eliminating $w_i$ leaves a generalised eigenproblem in $G$,
    solved from one SVD, as for :class:`~cca_zoo.linear.GCCA`, to which this
    reduces exactly when no view is missing. A sample seen in a single view
    still gets a latent score, and the views are tied together by the samples
    they share.

    Only whole views may be missing. Fill missing entries within an observed
    view first, for example with :class:`~sklearn.impute.IterativeImputer`
    through :class:`~cca_zoo.preprocessing.PerViewTransformer`.

    ``transform`` returns each view's scores, ``NaN`` for the samples that lack
    it. :meth:`transform_shared` gives every sample one score, the mean of the
    scores of the views it has.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view on the mean of its observed
            samples. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Per-view. Default is 0.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_observed_per_view_: Number of samples each view is observed in.

    Raises:
        ValueError: In ``fit``, if a row is partly missing, a sample has no
            view, or the views are not linked by at least
            ``n_components + 1`` shared samples, directly or through other
            views: without that, their scores are not tied to each other.

    References:
        Carroll, J. D. (1968). Generalization of canonical correlation analysis
        to three or more sets of variables. Proceedings of the 76th Annual
        Convention of the American Psychological Association, 3, 227-228.
        Kettenring, J. R. (1971). Canonical analysis of several sets of
        variables. Biometrika, 58(3), 433-451.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import IncompleteGCCA
        >>> rng = np.random.default_rng(0)
        >>> z = rng.standard_normal((60, 1))
        >>> X1 = z @ rng.standard_normal((1, 6)) + rng.standard_normal((60, 6))
        >>> X2 = z @ rng.standard_normal((1, 5)) + rng.standard_normal((60, 5))
        >>> X2[40:] = np.nan  # the last 20 samples have no second assay
        >>> model = IncompleteGCCA().fit([X1, X2])
        >>> model.transform_shared([X1, X2]).shape
        (60, 1)
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": RIDGE_PARAMETER,
    }

    _EPS: ClassVar[float] = 1e-6
    _supports_missing_views: ClassVar[bool] = True

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.0,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage

    def fit(self, views: list[ArrayLike], y: None = None) -> IncompleteGCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view,
                with the rows of samples that lack the view all ``NaN``.
            y: Ignored.

        Returns:
            self.
        """
        views_, observed = self._setup_missing(views)
        shrinkage = perview_parameter("shrinkage", self.shrinkage, 0.0, self.n_views_)

        # Eliminating the weights leaves max tr(G'QG) with G'DG = sI, where
        # Q = sum_i X_i C_i^{-1} X_i' / (n_i - 1) over the observed rows. It is
        # solved by the top singular vectors of the D^{-1/2}-scaled stack of
        # the whitened views, zero where a view is missing.
        counts = observed.sum(axis=1)
        whiteners, blocks = [], []
        for v, rows, c in zip(views_, observed.T, shrinkage):
            x = v[rows]
            n_i = len(x)
            cov = (1.0 - c) * (x.T @ x) / (n_i - 1) + c * np.eye(x.shape[1])
            whitener = psd_inverse_sqrt(cov, self._EPS)
            whiteners.append(whitener)
            block = np.zeros_like(v)
            block[rows] = x @ whitener / np.sqrt(n_i - 1)
            blocks.append(block)
        stack = np.concatenate(blocks, axis=1) / np.sqrt(counts)[:, None]
        U = np.linalg.svd(stack, full_matrices=False)[0][:, : self.n_components]
        # Unit-variance latent scores where every view is seen, as GCCA's.
        G = U * np.sqrt((self.n_samples_ - 1) * counts.mean() / counts)[:, None]

        self.weights_: list[np.ndarray] = [
            w @ (w @ (v[rows].T @ G[rows])) / (rows.sum() - 1)
            for v, rows, w in zip(views_, observed.T, whiteners)
        ]
        self.n_observed_per_view_: np.ndarray = observed.sum(axis=0)
        self._fit_maps_and_importances(views_)
        return self
