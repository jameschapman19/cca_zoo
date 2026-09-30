"""Gradient-boosted-tree CCA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, ClassVar

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    ey_grad_z,
    random_orthogonal_embedding,
)
from cca_zoo._utils._param_constraints import (
    FRACTION_PER_VIEW,
    NONNEGATIVE_PER_VIEW,
    POSITIVE_INT_PER_VIEW,
    RANDOM_STATE,
)
from cca_zoo._utils._validation import perview_parameter

try:
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    import catboost as cb

    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False


# Standard deviation of the random embedding the first round starts from, small
# against the unit scale of the EY optimum. It only breaks the symmetry (the EY
# gradient is zero at an all-zero embedding); later rounds use the trees alone.
_START_STD = 0.01


def _boosting_targets(representations: list[np.ndarray]) -> list[np.ndarray]:
    """Each sample's EY gradient, without the mean's ``4 / (M (n - 1))`` factor.

    Unscaled, the targets vanish at the optimum, so ``learning_rate`` is a
    true step size. ``float32``, as the XGBoost and LightGBM objectives need.
    """
    m, n = len(representations), representations[0].shape[0]
    return [
        (g * (m * (n - 1) / 4.0)).astype(np.float32) for g in ey_grad_z(representations)
    ]


class _XGBoostEncoder:
    """Per-view ensemble of ``k`` scalar XGBoost boosters, used during ``fit``."""

    def __init__(self, X: np.ndarray, k: int, params: dict[str, object]) -> None:
        self._params = params
        self._dtrain = xgb.DMatrix(X)
        self.boosters: list[xgb.Booster] = [
            xgb.train(params, self._dtrain, num_boost_round=0) for _ in range(k)
        ]

    def predict(self) -> np.ndarray:
        """Raw prediction on the training data, shape (n_samples, k)."""
        return np.column_stack(
            [b.predict(self._dtrain, output_margin=True) for b in self.boosters]
        )

    def boost(self, gradient: np.ndarray) -> None:
        """Add one tree to every component booster, fitted to ``gradient``."""
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


class _LightGBMEncoder:
    """Per-view ensemble of ``k`` scalar LightGBM boosters, used during ``fit``."""

    def __init__(self, X: np.ndarray, k: int, params: dict[str, object]) -> None:
        self._X = X
        dataset = lgb.Dataset(
            X, label=np.zeros(len(X), dtype=np.float32), params=params
        )
        self._dataset = dataset.construct()
        self.boosters: list[lgb.Booster] = [
            lgb.Booster(params=params, train_set=self._dataset) for _ in range(k)
        ]

    def predict(self) -> np.ndarray:
        """Raw prediction on the training data, shape (n_samples, k)."""
        return np.column_stack(
            [b.predict(self._X, raw_score=True) for b in self.boosters]
        )

    def boost(self, gradient: np.ndarray) -> None:
        """Add one tree to every component booster, fitted to ``gradient``."""
        for col, booster in enumerate(self.boosters):
            g = gradient[:, col].copy()

            def _fobj(
                _preds: np.ndarray, _train_set: object, _g: np.ndarray = g
            ) -> tuple[np.ndarray, np.ndarray]:
                return _g, np.ones_like(_g)

            booster.update(fobj=_fobj)


class _CatBoostGradientObjective:
    """CatBoost custom loss relaying a fixed gradient with a unit Hessian.

    CatBoost expects negated derivatives: ``der1 = -gradient``, ``der2 = -1``.
    """

    def __init__(self, gradient: np.ndarray) -> None:
        self._gradient = gradient

    def calc_ders_range(
        self,
        approxes: list[float],
        targets: list[float],
        weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        return [(-float(g), -1.0) for g in self._gradient]


class _CatBoostEncoder:
    """Per-view ensemble of ``k`` scalar CatBoost boosters, used during ``fit``.

    CatBoost cannot add a tree in place, so each round replaces every booster
    with one continued from it via ``init_model``.
    """

    def __init__(self, X: np.ndarray, k: int, params: dict[str, object]) -> None:
        self._X = X
        self._params = params
        # CatBoost refuses to train on a label vector that is all one value;
        # the custom objective ignores it entirely, so any two-valued vector
        # satisfies the check without needing real labels.
        self._y_dummy = np.resize([0.0, 1.0], len(X)).astype(np.float32)
        self.boosters: list[Any] = [None] * k

    def predict(self) -> np.ndarray:
        """Raw prediction on the training data, shape (n_samples, k)."""
        n = len(self._X)
        return np.column_stack(
            [
                np.zeros(n, dtype=np.float32)
                if booster is None
                else np.asarray(
                    booster.predict(self._X, prediction_type="RawFormulaVal"),
                    dtype=np.float32,
                )
                for booster in self.boosters
            ]
        )

    def boost(self, gradient: np.ndarray) -> None:
        """Add one tree to every component booster, fitted to ``gradient``."""
        updated = []
        for col, booster in enumerate(self.boosters):
            objective = _CatBoostGradientObjective(gradient[:, col])
            new_model = cb.CatBoostRegressor(
                iterations=1, loss_function=objective, **self._params
            )
            new_model.fit(self._X, self._y_dummy, init_model=booster, verbose=False)
            updated.append(new_model)
        self.boosters = updated


class TreeCCA(BaseModel, ABC):
    r"""Base class for nonlinear CCA with gradient-boosted-tree encoders.

    Each view's encoder is one boosted ensemble per latent dimension, fitted
    to minimise the EY loss (:mod:`cca_zoo._utils._ey`)

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V).
    $$

    Each round adds one tree per component, fitted to each sample's EY
    gradient, visiting the views in turn. The gradient vanishes at zero, so
    the first round starts from a small random embedding; the encoders are the
    trees alone. Use a backend
    subclass: :class:`XGBoostCCA`, :class:`LightGBMCCA` or
    :class:`CatBoostCCA`.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        n_estimators: Boosting rounds; a view whose budget runs out stops
            changing. Per-view. Default is 200.
        max_depth: Maximum tree depth. Per-view. Default is 3.
        learning_rate: Boosting step size. Per-view. Default is 0.1.
        subsample: Row subsampling ratio per tree. Per-view. Default is 0.8.
        colsample_bytree: Column subsampling ratio per tree. Per-view.
            Default is 0.8.
        min_child_weight: Minimum hessian (XGBoost) or samples (LightGBM,
            CatBoost) per leaf. Per-view. Default is 20.
        gauss_seidel: Whether to recompute the gradient after each view's
            update (Gauss-Seidel) rather than once per round (Jacobi).
            Default is True.
        random_state: Seed for the boosters and the initial embedding.
            Default is None.

    Attributes:
        boosters_: Per view, one fitted booster per latent dimension.

    References:
        Chapman, J. (2026). TreeCCA: Canonical Correlation Analysis via
        Gradient-Boosted Trees. arXiv:2607.27027.
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "n_estimators": POSITIVE_INT_PER_VIEW,
        "max_depth": POSITIVE_INT_PER_VIEW,
        "learning_rate": [Interval(Real, 0, None, closed="neither"), "array-like"],
        "subsample": FRACTION_PER_VIEW,
        "colsample_bytree": FRACTION_PER_VIEW,
        "min_child_weight": NONNEGATIVE_PER_VIEW,
        "gauss_seidel": ["boolean"],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        n_estimators: int | list[int] = 200,
        max_depth: int | list[int] = 3,
        learning_rate: float | list[float] = 0.1,
        subsample: float | list[float] = 0.8,
        colsample_bytree: float | list[float] = 0.8,
        min_child_weight: float | list[float] = 20,
        gauss_seidel: bool = True,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gauss_seidel = gauss_seidel
        self.random_state = random_state

    @abstractmethod
    def _booster_params(
        self,
        learning_rate: float,
        max_depth: int,
        subsample: float,
        colsample_bytree: float,
        min_child_weight: float,
        seed: int,
    ) -> dict[str, object]:
        """Backend training parameters for one view, from its resolved settings."""

    @abstractmethod
    def _make_encoder(
        self, X: np.ndarray, k: int, params: dict[str, object]
    ) -> _XGBoostEncoder | _LightGBMEncoder | _CatBoostEncoder:
        """A zero-tree encoder of ``k`` boosters for one view's training data."""

    @abstractmethod
    def _predict_boosters(self, boosters: list[Any], X: np.ndarray) -> np.ndarray:
        """Backend prediction of fitted boosters on new data, shape (n_samples, k)."""

    @abstractmethod
    def _booster_gain(self, booster: Any, n_features: int) -> np.ndarray:
        """Total split gain per feature of one fitted booster, shape (n_features,)."""

    def _feature_importances(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """Total split gain over each view's boosters."""
        return [
            np.sum([self._booster_gain(b, p) for b in boosters], axis=0)
            for boosters, p in zip(self.boosters_, self.n_features_per_view_)
        ]

    def fit(self, views: list[ArrayLike], y: None = None) -> TreeCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        k = self.n_components
        n_views = len(views_)

        n_estimators_ = perview_parameter(
            "n_estimators", self.n_estimators, 200, n_views
        )
        max_depth_ = perview_parameter("max_depth", self.max_depth, 3, n_views)
        learning_rate_ = perview_parameter(
            "learning_rate", self.learning_rate, 0.1, n_views
        )
        subsample_ = perview_parameter("subsample", self.subsample, 0.8, n_views)
        colsample_bytree_ = perview_parameter(
            "colsample_bytree", self.colsample_bytree, 0.8, n_views
        )
        min_child_weight_ = perview_parameter(
            "min_child_weight", self.min_child_weight, 20, n_views
        )

        rng = np.random.default_rng(self.random_state)
        starts = [
            random_orthogonal_embedding(X, k, rng, std=_START_STD) for X in views_
        ]
        # The backends take an integer seed; drawing it from the fit's own
        # generator makes random_state=None give a fresh one each fit.
        seed = int(rng.integers(2**31 - 1))

        params_per_view = [
            self._booster_params(
                learning_rate_[i],
                max_depth_[i],
                subsample_[i],
                colsample_bytree_[i],
                min_child_weight_[i],
                seed,
            )
            for i in range(n_views)
        ]
        encoders = [
            self._make_encoder(X, k, p) for X, p in zip(views_, params_per_view)
        ]

        for round_idx in range(max(n_estimators_)):
            offsets = starts if round_idx == 0 else [np.float32(0.0)] * n_views
            representations = [o + enc.predict() for o, enc in zip(offsets, encoders)]
            grads = _boosting_targets(representations)

            for view_idx in range(n_views):
                if round_idx >= n_estimators_[view_idx]:
                    continue
                encoders[view_idx].boost(grads[view_idx])
                if self.gauss_seidel and view_idx < n_views - 1:
                    representations[view_idx] = (
                        offsets[view_idx] + encoders[view_idx].predict()
                    )
                    grads = _boosting_targets(representations)

        self.boosters_: list[list[Any]] = [enc.boosters for enc in encoders]
        # The EY loss sees only covariances, so the boosted scores carry an
        # arbitrary offset: centre them on the training data.
        self._score_means_: list[np.ndarray] = [
            self._predict_boosters(b, X).astype(np.float64).mean(axis=0)
            for b, X in zip(self.boosters_, views_)
        ]
        self._fit_maps_and_importances(views_)
        return self

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        scores: np.ndarray = (
            self._predict_boosters(self.boosters_[view], centred).astype(np.float64)
            - self._score_means_[view]
        )
        return scores


class XGBoostCCA(TreeCCA):
    """:class:`TreeCCA` with XGBoost boosters.

    Parameters and attributes are those of :class:`TreeCCA`.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.tree import XGBoostCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 5))
        >>> X2 = rng.standard_normal((100, 5))
        >>> model = XGBoostCCA(
        ...     n_components=2, n_estimators=[10, 20], max_depth=[3, 6]
        ... ).fit([X1, X2])
        >>> Z1, Z2 = model.transform([X1, X2])
    """

    def _booster_params(
        self,
        learning_rate: float,
        max_depth: int,
        subsample: float,
        colsample_bytree: float,
        min_child_weight: float,
        seed: int,
    ) -> dict[str, object]:
        return {
            "tree_method": "hist",
            "base_score": 0.0,
            "disable_default_eval_metric": True,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "seed": seed,
        }

    def _make_encoder(
        self, X: np.ndarray, k: int, params: dict[str, object]
    ) -> _XGBoostEncoder:
        return _XGBoostEncoder(X, k, params)

    def _predict_boosters(self, boosters: list[Any], X: np.ndarray) -> np.ndarray:
        dmatrix = xgb.DMatrix(X)
        return np.column_stack(
            [b.predict(dmatrix, output_margin=True) for b in boosters]
        )

    def _booster_gain(self, booster: Any, n_features: int) -> np.ndarray:
        gain = booster.get_score(importance_type="total_gain")
        return np.array([gain.get(f"f{j}", 0.0) for j in range(n_features)])


class LightGBMCCA(TreeCCA):
    """:class:`TreeCCA` with LightGBM boosters.

    Parameters and attributes are those of :class:`TreeCCA`. Requires
    ``lightgbm``, in the ``tree`` extra.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.tree import LightGBMCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 5))
        >>> X2 = rng.standard_normal((100, 5))
        >>> model = LightGBMCCA(n_components=2, n_estimators=10).fit([X1, X2])
    """

    def fit(self, views: list[ArrayLike], y: None = None) -> LightGBMCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ImportError: If ``lightgbm`` is not installed.
        """
        if not _LGBM_AVAILABLE:
            raise ImportError(
                "LightGBMCCA requires the lightgbm package. "
                "Install with: pip install lightgbm"
            )
        return super().fit(views, y)

    def _booster_params(
        self,
        learning_rate: float,
        max_depth: int,
        subsample: float,
        colsample_bytree: float,
        min_child_weight: float,
        seed: int,
    ) -> dict[str, object]:
        return {
            "objective": "regression",
            "metric": "None",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "feature_fraction": colsample_bytree,
            "bagging_fraction": subsample,
            "bagging_freq": 1,
            "min_child_samples": int(min_child_weight),
            # Keep features LightGBM would pre-filter as unsplittable at this
            # leaf size: on small data it otherwise drops every feature and
            # refuses to train, where the other backends fit without splits.
            "feature_pre_filter": False,
            "min_data_in_bin": 1,
            "verbose": -1,
            "seed": seed,
        }

    def _make_encoder(
        self, X: np.ndarray, k: int, params: dict[str, object]
    ) -> _LightGBMEncoder:
        return _LightGBMEncoder(X, k, params)

    def _predict_boosters(self, boosters: list[Any], X: np.ndarray) -> np.ndarray:
        return np.column_stack([b.predict(X, raw_score=True) for b in boosters])

    def _booster_gain(self, booster: Any, n_features: int) -> np.ndarray:
        return np.asarray(booster.feature_importance(importance_type="gain"), float)


class CatBoostCCA(TreeCCA):
    """:class:`TreeCCA` with CatBoost boosters.

    Parameters and attributes are those of :class:`TreeCCA`. Requires
    ``catboost``, in the ``tree`` extra. Slower per round than the other
    backends, since CatBoost rebuilds each booster to add a tree.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.tree import CatBoostCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 5))
        >>> X2 = rng.standard_normal((100, 5))
        >>> model = CatBoostCCA(n_components=2, n_estimators=10).fit([X1, X2])
    """

    def fit(self, views: list[ArrayLike], y: None = None) -> CatBoostCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ImportError: If ``catboost`` is not installed.
        """
        if not _CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoostCCA requires the catboost package. "
                "Install with: pip install catboost"
            )
        return super().fit(views, y)

    def _booster_params(
        self,
        learning_rate: float,
        max_depth: int,
        subsample: float,
        colsample_bytree: float,
        min_child_weight: float,
        seed: int,
    ) -> dict[str, object]:
        return {
            "depth": max_depth,
            # Depthwise is the only grow policy that honours min_data_in_leaf,
            # which the default symmetric trees ignore; the other backends'
            # minimum leaf size needs it to mean the same thing.
            "grow_policy": "Depthwise",
            "learning_rate": learning_rate,
            "subsample": subsample,
            "bootstrap_type": "Bernoulli",
            "rsm": colsample_bytree,
            "min_data_in_leaf": int(min_child_weight),
            "boost_from_average": False,
            "eval_metric": "RMSE",
            "allow_writing_files": False,
            "verbose": False,
            "random_seed": seed,
        }

    def _make_encoder(
        self, X: np.ndarray, k: int, params: dict[str, object]
    ) -> _CatBoostEncoder:
        return _CatBoostEncoder(X, k, params)

    def _predict_boosters(self, boosters: list[Any], X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack(
            [
                np.zeros(n, dtype=np.float32)
                if booster is None
                else np.asarray(
                    booster.predict(X, prediction_type="RawFormulaVal"),
                    dtype=np.float32,
                )
                for booster in boosters
            ]
        )

    def _booster_gain(self, booster: Any, n_features: int) -> np.ndarray:
        if booster is None:
            return np.zeros(n_features)
        return np.asarray(booster.get_feature_importance(), float)
