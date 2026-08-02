"""TreeCCA — gradient-boosted-tree Canonical Correlation Analysis."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._validation import validate_views


def _ey_grad(
    Z1: np.ndarray, Z2: np.ndarray, target_std: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Eckart-Young (EY) loss gradient w.r.t. two centred embeddings.

    Args:
        Z1: View-1 embedding, shape (n_samples, k).
        Z2: View-2 embedding, shape (n_samples, k).
        target_std: Target standard deviation used to rescale the raw
            gradient so that boosted-tree leaf values stay well-conditioned.

    Returns:
        Tuple ``(G1, G2)`` of gradients, each shape (n_samples, k) and
        dtype ``float32`` (required by XGBoost custom objectives).
    """
    n = Z1.shape[0]
    Z1c = Z1 - Z1.mean(axis=0)
    Z2c = Z2 - Z2.mean(axis=0)
    V = (Z1c.T @ Z1c + Z2c.T @ Z2c) / (n - 1)
    G1_raw = -Z2c + Z1c @ V
    G2_raw = -Z1c + Z2c @ V
    scale = max(float(G1_raw.std()), float(G2_raw.std()), 1e-6)
    G1 = (G1_raw / scale * target_std).astype(np.float32)
    G2 = (G2_raw / scale * target_std).astype(np.float32)
    return G1, G2


def _pca_base_margin(Xc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Whitened-PCA initial embedding and its reusable linear projection.

    Gives each of the ``k`` initial components unit variance and zero
    cross-component covariance, so every boosted-tree component receives
    equal, non-vanishing EY-gradient signal from round zero.

    Args:
        Xc: Mean-centred training view, shape (n_samples, n_features).
        k: Number of components.

    Returns:
        Tuple ``(base_margin, projection)``: ``base_margin`` has shape
        (n_samples, k) and is the initial embedding for training;
        ``projection`` has shape (n_features, k) and reproduces the same
        whitened embedding for unseen data via ``Xc_new @ projection``.
    """
    n = Xc.shape[0]
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    projection = (Vt[:k].T / s[:k] * np.sqrt(n - 1)).astype(np.float32)
    base_margin = (Xc @ projection).astype(np.float32)
    return base_margin, projection


class TreeCCA(BaseModel):
    r"""TreeCCA — nonlinear two-view CCA with gradient-boosted-tree encoders.

    Learns two nonlinear encoders :math:`f_1, f_2` (one gradient-boosted tree
    ensemble per latent dimension, per view) that maximise the Eckart-Young
    (EY) unconstrained-CCA objective:

    .. math::

        \mathcal{L}_{EY}(Z_1, Z_2) = 2 \operatorname{tr}(C_{12})
            - \operatorname{tr}(V_{11} + V_{22})

    where :math:`Z_i = f_i(X_i)`, :math:`C_{12}` is the cross-covariance and
    :math:`V_{ii}` the within-view covariance of the centred embeddings. The
    encoders are fit by alternating (Gauss-Seidel) gradient boosting: each
    round, one XGBoost tree is added to each of the :math:`2k` boosters
    (:math:`k` = ``latent_dimensions``) using the EY-loss gradient as a
    custom regression objective. Because each latent component is a
    boosted-tree ensemble, per-component feature importance (split gain) is
    available directly, without a separate interpretability method such as
    SHAP.

    This is a from-scratch reimplementation, as a scikit-learn-style
    :class:`~cca_zoo._base.BaseModel`, of the "Design A" (sequential,
    scalar-booster) training procedure from the TreeCCA research codebase.
    Only two views are currently supported.

    References:
        Chapman, J., Wells, L., & Lawry Aguila, L. (2024). Unconstrained
        stochastic CCA: Unifying multiview and self-supervised learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        n_estimators: Number of boosting rounds (trees added per booster).
            Default is 50.
        max_depth: Maximum depth of each tree. Default is 5.
        learning_rate: Boosting learning rate (XGBoost ``eta``).
            Default is 0.1.
        subsample: Row subsampling ratio per tree. Default is 0.8.
        colsample_bytree: Column subsampling ratio per tree. Default is 0.8.
        min_child_weight: Minimum sum of instance weight needed in a child.
            Default is 5.
        gauss_seidel: If True, re-predict view 1's embedding after updating
            its boosters and use the fresh values when computing view 2's
            gradient (Gauss-Seidel); if False, both gradients are computed
            from the same stale embeddings (Jacobi). Default is True.
        random_state: Seed for the XGBoost boosters. Default is 0.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 5))
        >>> X2 = rng.standard_normal((100, 5))
        >>> model = TreeCCA(latent_dimensions=2, n_estimators=10).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        n_estimators: int = 50,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 5,
        gauss_seidel: bool = True,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gauss_seidel = gauss_seidel
        self.random_state = random_state

    def _xgb_params(self) -> dict[str, object]:
        """Build the XGBoost parameter dictionary for a scalar booster.

        Returns:
            Dictionary of XGBoost training parameters.
        """
        return {
            "tree_method": "hist",
            "base_score": 0.0,
            "disable_default_eval_metric": True,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "seed": int(self.random_state),
        }

    @staticmethod
    def _predict_margin(
        boosters: list[xgb.Booster], dmatrices: list[xgb.DMatrix]
    ) -> np.ndarray:
        """Predict the raw (base-margin-inclusive) embedding for one view.

        Args:
            boosters: One booster per latent component.
            dmatrices: One DMatrix per latent component (same view).

        Returns:
            Array of shape (n_samples, k).
        """
        return np.column_stack(
            [b.predict(d, output_margin=True) for b, d in zip(boosters, dmatrices)]
        )

    def fit(self, views: list[ArrayLike], y: None = None) -> TreeCCA:
        """Fit the TreeCCA model.

        Args:
            views: List of exactly two arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If a number of views other than two is provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_ = self._setup_fit(views)
        if len(views_) != 2:
            raise ValueError(
                f"TreeCCA currently supports exactly two views, got {len(views_)}."
            )
        k = self.latent_dimensions
        X1, X2 = views_

        bm1, proj1 = _pca_base_margin(X1, k)
        bm2, proj2 = _pca_base_margin(X2, k)
        self._projections_: list[np.ndarray] = [proj1, proj2]

        params = self._xgb_params()

        def _make_dmatrices(X: np.ndarray, bm: np.ndarray) -> list[xgb.DMatrix]:
            dmatrices = []
            for col in range(k):
                dm = xgb.DMatrix(X)
                dm.set_base_margin(bm[:, col])
                dmatrices.append(dm)
            return dmatrices

        dm1 = _make_dmatrices(X1, bm1)
        dm2 = _make_dmatrices(X2, bm2)

        bst1 = [xgb.train(params, dm1[col], num_boost_round=0) for col in range(k)]
        bst2 = [xgb.train(params, dm2[col], num_boost_round=0) for col in range(k)]

        for _ in range(self.n_estimators):
            Z1 = self._predict_margin(bst1, dm1)
            Z2 = self._predict_margin(bst2, dm2)
            G1, G2 = _ey_grad(Z1, Z2)

            bst1 = self._boost_one_round(bst1, dm1, G1, params)

            if self.gauss_seidel:
                Z1 = self._predict_margin(bst1, dm1)
                _, G2 = _ey_grad(Z1, Z2)

            bst2 = self._boost_one_round(bst2, dm2, G2, params)

        self.boosters_: list[list[xgb.Booster]] = [bst1, bst2]
        return self

    @staticmethod
    def _boost_one_round(
        boosters: list[xgb.Booster],
        dmatrices: list[xgb.DMatrix],
        gradient: np.ndarray,
        params: dict[str, object],
    ) -> list[xgb.Booster]:
        """Add one tree to each component booster using the EY gradient.

        Args:
            boosters: Current boosters, one per latent component.
            dmatrices: DMatrices, one per latent component (same view).
            gradient: EY gradient for this view, shape (n_samples, k).
            params: XGBoost training parameters.

        Returns:
            Updated list of boosters.
        """
        updated = []
        for col, (booster, dmatrix) in enumerate(zip(boosters, dmatrices)):
            g = gradient[:, col].copy()

            def _objective(
                _predt: np.ndarray, _dtrain: xgb.DMatrix, _g: np.ndarray = g
            ) -> tuple[np.ndarray, np.ndarray]:
                return _g, np.ones_like(_g)

            updated.append(
                xgb.train(
                    params,
                    dmatrix,
                    num_boost_round=1,
                    obj=_objective,
                    xgb_model=booster,
                )
            )
        return updated

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project views into the latent space using the fitted boosters.

        Args:
            views: List of exactly two arrays, each (n_samples, n_features_i).

        Returns:
            List of two arrays, each (n_samples, latent_dimensions).

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            ValueError: If a number of views other than two is provided.
        """
        check_is_fitted(self)
        validated = validate_views(views)
        if len(validated) != 2:
            raise ValueError(
                f"TreeCCA currently supports exactly two views, got {len(validated)}."
            )
        centred = [v - m for v, m in zip(validated, self.means_)]
        result = []
        for v, boosters, projection in zip(centred, self.boosters_, self._projections_):
            bm = v @ projection
            cols = []
            for col, booster in enumerate(boosters):
                dm = xgb.DMatrix(v)
                dm.set_base_margin(bm[:, col])
                cols.append(booster.predict(dm, output_margin=True))
            result.append(np.column_stack(cols))
        return result

    @property
    def weights(self) -> list[np.ndarray]:
        """Not implemented for TreeCCA.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: TreeCCA encoders are boosted-tree ensembles,
                not linear weight matrices. Use ``boosters_`` instead, e.g.
                ``model.boosters_[view][component].get_score(importance_type="gain")``
                for per-component feature importance.
        """
        check_is_fitted(self)
        raise NotImplementedError(
            "TreeCCA has no linear weight matrices; its encoders are "
            "gradient-boosted-tree ensembles. Use the `boosters_` attribute "
            "instead, e.g. model.boosters_[view][component]"
            '.get_score(importance_type="gain") for per-component feature '
            "importance."
        )
