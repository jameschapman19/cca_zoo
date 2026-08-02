"""TreeCCA — gradient-boosted-tree Canonical Correlation Analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import ey_grad_z
from cca_zoo._utils._validation import validate_views

try:
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


def _rescale_to_target_std(
    grads: list[np.ndarray], target_std: float = 0.1
) -> list[np.ndarray]:
    """Rescale a set of per-view gradients to a common target standard deviation.

    Boosted-tree leaf values are well-conditioned only for a roughly-fixed
    gradient scale, so the exact (analytic) EY gradient is rescaled by a
    single shared scalar before being used as a custom-objective target.
    Since the same scalar is applied to every view, this changes only the
    effective step size, not the gradient's direction or relative
    cross-view magnitudes.

    Args:
        grads: One gradient array per view, each (n_samples, k).
        target_std: Target standard deviation. Default is 0.1.

    Returns:
        List of rescaled gradients, dtype ``float32`` (required for
        XGBoost/LightGBM custom objectives).
    """
    scale = max(max(float(g.std()) for g in grads), 1e-6)
    return [(g / scale * target_std).astype(np.float32) for g in grads]


def _random_orthogonal_base_margin(
    Xc: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-variance random-orthogonal initial embedding and its projection.

    Draws ``k`` random orthogonal directions in feature space (independent of
    the data's principal directions) and rescales them so each initial
    component has unit variance. The unit-variance scaling is what matters
    for a well-conditioned, non-vanishing EY gradient from round zero;
    orthogonality keeps the initial cross-component covariance at zero.

    Args:
        Xc: Mean-centred training view, shape (n_samples, n_features).
        k: Number of components. Must not exceed ``n_features``.
        rng: Random generator used to draw the orthogonal directions.

    Returns:
        Tuple ``(base_margin, projection)``: ``base_margin`` has shape
        (n_samples, k) and is the initial embedding for training;
        ``projection`` has shape (n_features, k) and reproduces the same
        unit-variance embedding for unseen data via ``Xc_new @ projection``.
    """
    n, p = Xc.shape
    W, _ = np.linalg.qr(rng.standard_normal((p, k)))
    Z = Xc @ W
    scale = np.linalg.norm(Z, axis=0, keepdims=True) / np.sqrt(n - 1)
    projection = (W / scale).astype(np.float32)
    base_margin = (Z / scale).astype(np.float32)
    return base_margin, projection


class _Encoder:
    """Per-view ensemble of ``k`` scalar boosters, used only during ``fit``.

    Dispatches to XGBoost or LightGBM depending on ``backend``. Predictions
    are the raw sum of tree outputs (no base margin / init score); the
    caller is responsible for adding the fixed initial embedding.
    """

    def __init__(
        self, backend: str, X: np.ndarray, k: int, params: dict[str, object]
    ) -> None:
        self.backend = backend
        self._X = X
        self._params = params
        if backend == "xgboost":
            self._dtrain = xgb.DMatrix(X)
            self.boosters: list[Any] = [
                xgb.train(params, self._dtrain, num_boost_round=0) for _ in range(k)
            ]
        else:
            dataset = lgb.Dataset(
                X, label=np.zeros(len(X), dtype=np.float32), params=params
            )
            self._dataset = dataset.construct()
            self.boosters = [
                lgb.Booster(params=params, train_set=self._dataset) for _ in range(k)
            ]

    def predict(self) -> np.ndarray:
        """Raw (base-margin-free) prediction on the training data.

        Returns:
            Array of shape (n_samples, k).
        """
        if self.backend == "xgboost":
            return np.column_stack(
                [b.predict(self._dtrain, output_margin=True) for b in self.boosters]
            )
        return np.column_stack(
            [b.predict(self._X, raw_score=True) for b in self.boosters]
        )

    def boost(self, gradient: np.ndarray) -> None:
        """Add one tree to every component booster using the EY gradient.

        Args:
            gradient: EY gradient for this view, shape (n_samples, k).
        """
        if self.backend == "xgboost":
            updated = []
            for col, booster in enumerate(self.boosters):
                g = gradient[:, col].copy()

                def _objective(
                    _predt: np.ndarray, _dtrain: xgb.DMatrix, _g: np.ndarray = g
                ) -> tuple[np.ndarray, np.ndarray]:
                    return _g, np.ones_like(_g)

                updated.append(
                    xgb.train(
                        self._params,
                        self._dtrain,
                        num_boost_round=1,
                        obj=_objective,
                        xgb_model=booster,
                    )
                )
            self.boosters = updated
        else:
            for col, booster in enumerate(self.boosters):
                g = gradient[:, col].copy()

                def _fobj(
                    _preds: np.ndarray, _train_set: object, _g: np.ndarray = g
                ) -> tuple[np.ndarray, np.ndarray]:
                    return _g, np.ones_like(_g)

                booster.update(fobj=_fobj)


class TreeCCA(BaseModel):
    r"""TreeCCA — nonlinear multiview CCA with gradient-boosted-tree encoders.

    Learns one nonlinear encoder :math:`f_i` per view (a gradient-boosted
    tree ensemble per latent dimension) that jointly maximise the
    Eckart-Young (EY) unconstrained-CCA objective:

    .. math::

        \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)

    where, for embeddings :math:`Z_i = f_i(X_i)`, :math:`C` is the mean
    pairwise cross-covariance (including :math:`i = j` terms) and :math:`V`
    the mean auto-covariance across all views (see
    :mod:`cca_zoo._utils._ey`, the same shared EY-loss machinery used by
    :class:`~cca_zoo.linear.gradient.CCA_EY` and
    :class:`~cca_zoo.deep.DCCA_EY`). The encoders are fit by alternating
    (Gauss-Seidel) gradient boosting: each round, for every view in turn, one
    tree is added to each of its ``latent_dimensions`` boosters using the
    EY-loss gradient (rescaled to a fixed target standard deviation for
    well-conditioned tree leaves) as a custom regression objective, and —
    when ``gauss_seidel=True`` — the gradient is recomputed from the
    freshest embeddings before moving to the next view. Training starts from
    a random-orthogonal, unit-variance initial embedding per view. Because
    each latent component is a boosted-tree ensemble, per-component feature
    importance (split gain) is available directly, without a separate
    interpretability method such as SHAP.

    This is a from-scratch reimplementation, as a scikit-learn-style
    :class:`~cca_zoo._base.BaseModel`, of the "Design A" (sequential,
    scalar-booster) training procedure from the TreeCCA research codebase,
    generalised from two views to an arbitrary number of views.

    References:
        Chapman, J., Wells, L., & Lawry Aguila, L. (2024). Unconstrained
        stochastic CCA: Unifying multiview and self-supervised learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        backend: Gradient-boosting library used for the per-component
            encoders: ``"xgboost"`` (default) or ``"lightgbm"``. The
            ``"lightgbm"`` backend requires the optional ``lightgbm``
            package.
        n_estimators: Number of boosting rounds (trees added per booster).
            Default is 50.
        max_depth: Maximum depth of each tree. Default is 5.
        learning_rate: Boosting learning rate. Default is 0.1.
        subsample: Row subsampling ratio per tree. Default is 0.8.
        colsample_bytree: Column subsampling ratio per tree. Default is 0.8.
        min_child_weight: Minimum sum of instance weight (xgboost) / minimum
            number of samples (lightgbm) needed in a child. Default is 5.
        gauss_seidel: If True, re-predict view 1's embedding after updating
            its boosters and use the fresh values when computing view 2's
            gradient (Gauss-Seidel); if False, both gradients are computed
            from the same stale embeddings (Jacobi). Default is True.
        random_state: Seed for the boosters and for drawing the
            random-orthogonal initial embedding. Default is 0.

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
        backend: str = "xgboost",
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
        self.backend = backend
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gauss_seidel = gauss_seidel
        self.random_state = random_state

    def _booster_params(self) -> dict[str, object]:
        """Build the per-backend booster parameter dictionary.

        Returns:
            Dictionary of training parameters for the selected backend.
        """
        if self.backend == "xgboost":
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
        return {
            "objective": "regression",
            "metric": "None",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "feature_fraction": self.colsample_bytree,
            "bagging_fraction": self.subsample,
            "bagging_freq": 1,
            "min_child_samples": int(self.min_child_weight),
            "min_data_in_bin": 1,
            "verbose": -1,
            "seed": int(self.random_state),
        }

    def _predict_boosters(self, boosters: list[Any], X: np.ndarray) -> np.ndarray:
        """Raw (base-margin-free) prediction for arbitrary (e.g. test) data.

        Args:
            boosters: One fitted booster per latent component.
            X: Input array, shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, k).
        """
        if self.backend == "xgboost":
            dmatrix = xgb.DMatrix(X)
            return np.column_stack(
                [b.predict(dmatrix, output_margin=True) for b in boosters]
            )
        return np.column_stack([b.predict(X, raw_score=True) for b in boosters])

    def fit(self, views: list[ArrayLike], y: None = None) -> TreeCCA:
        """Fit the TreeCCA model.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
            ValueError: If ``backend`` is not ``"xgboost"`` or ``"lightgbm"``.
            ImportError: If ``backend="lightgbm"`` but lightgbm is not
                installed.
        """
        if self.backend not in ("xgboost", "lightgbm"):
            raise ValueError(
                f"backend must be 'xgboost' or 'lightgbm', got {self.backend!r}."
            )
        if self.backend == "lightgbm" and not _LGBM_AVAILABLE:
            raise ImportError(
                "backend='lightgbm' requires the lightgbm package. "
                "Install with: pip install lightgbm"
            )
        views_ = self._setup_fit(views)
        k = self.latent_dimensions
        n_views = len(views_)

        rng = np.random.default_rng(self.random_state)
        base_margins = []
        projections = []
        for X in views_:
            bm, proj = _random_orthogonal_base_margin(X, k, rng)
            base_margins.append(bm)
            projections.append(proj)
        self._projections_: list[np.ndarray] = projections

        params = self._booster_params()
        encoders = [_Encoder(self.backend, X, k, params) for X in views_]

        for _ in range(self.n_estimators):
            representations = [
                bm + enc.predict() for bm, enc in zip(base_margins, encoders)
            ]
            grads = _rescale_to_target_std(ey_grad_z(representations))

            for view_idx in range(n_views):
                encoders[view_idx].boost(grads[view_idx])
                if self.gauss_seidel and view_idx < n_views - 1:
                    representations[view_idx] = (
                        base_margins[view_idx] + encoders[view_idx].predict()
                    )
                    grads = _rescale_to_target_std(ey_grad_z(representations))

        self.boosters_: list[list[Any]] = [enc.boosters for enc in encoders]
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project views into the latent space using the fitted boosters.

        Args:
            views: List of arrays, each (n_samples, n_features_i), matching
                the number of views passed to ``fit``.

        Returns:
            List of arrays, each (n_samples, latent_dimensions).

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            ValueError: If fewer than 2 views are provided.
        """
        check_is_fitted(self)
        validated = validate_views(views)
        centred = [v - m for v, m in zip(validated, self.means_)]
        result = []
        for v, boosters, projection in zip(centred, self.boosters_, self._projections_):
            bm = v @ projection
            result.append(bm + self._predict_boosters(boosters, v))
        return result

    @property
    def weights(self) -> list[np.ndarray]:
        """Not implemented for TreeCCA.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: TreeCCA encoders are boosted-tree ensembles,
                not linear weight matrices. Use ``boosters_`` instead, e.g.
                ``model.boosters_[view][component].get_score(importance_type="gain")``
                (xgboost backend) or
                ``model.boosters_[view][component].feature_importance(importance_type="gain")``
                (lightgbm backend) for per-component feature importance.
        """
        check_is_fitted(self)
        raise NotImplementedError(
            "TreeCCA has no linear weight matrices; its encoders are "
            "gradient-boosted-tree ensembles. Use the `boosters_` attribute "
            "instead for per-component feature importance, e.g. "
            'model.boosters_[view][component].get_score(importance_type="gain") '
            "for the xgboost backend, or "
            "model.boosters_[view][component]"
            '.feature_importance(importance_type="gain") for the lightgbm '
            "backend."
        )
