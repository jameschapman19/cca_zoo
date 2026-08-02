"""TreeCCA — gradient-boosted-tree Canonical Correlation Analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._validation import validate_views

try:
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


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
        dtype ``float32`` (required as custom-objective gradients).
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
    round, one tree is added to each of the :math:`2k` boosters
    (:math:`k` = ``latent_dimensions``) using the EY-loss gradient as a
    custom regression objective, starting from a random-orthogonal,
    unit-variance initial embedding. Because each latent component is a
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
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in either view. Default is 1.
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
            views: List of exactly two arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If a number of views other than two is provided.
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
        if len(views_) != 2:
            raise ValueError(
                f"TreeCCA currently supports exactly two views, got {len(views_)}."
            )
        k = self.latent_dimensions
        X1, X2 = views_

        rng = np.random.default_rng(self.random_state)
        bm1, proj1 = _random_orthogonal_base_margin(X1, k, rng)
        bm2, proj2 = _random_orthogonal_base_margin(X2, k, rng)
        self._projections_: list[np.ndarray] = [proj1, proj2]

        params = self._booster_params()
        encoder1 = _Encoder(self.backend, X1, k, params)
        encoder2 = _Encoder(self.backend, X2, k, params)

        for _ in range(self.n_estimators):
            Z1 = bm1 + encoder1.predict()
            Z2 = bm2 + encoder2.predict()
            G1, G2 = _ey_grad(Z1, Z2)

            encoder1.boost(G1)

            if self.gauss_seidel:
                Z1 = bm1 + encoder1.predict()
                _, G2 = _ey_grad(Z1, Z2)

            encoder2.boost(G2)

        self.boosters_: list[list[Any]] = [encoder1.boosters, encoder2.boosters]
        return self

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
