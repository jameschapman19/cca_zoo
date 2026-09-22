"""TreeCCA — gradient-boosted-tree Canonical Correlation Analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    ey_grad_z,
    random_orthogonal_embedding,
    rescale_grads_to_target_std,
)
from cca_zoo._utils._validation import perview_parameter, validate_views

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


def _rescale_to_target_std(
    grads: list[np.ndarray], target_std: float = 0.1
) -> list[np.ndarray]:
    """As :func:`cca_zoo._utils._ey.rescale_grads_to_target_std`, cast to float32.

    ``float32`` is required for XGBoost/LightGBM custom objectives.

    Args:
        grads: One gradient array per view, each (n_samples, k).
        target_std: Target standard deviation. Default is 0.1.

    Returns:
        List of rescaled gradients, dtype ``float32``.
    """
    return [
        g.astype(np.float32) for g in rescale_grads_to_target_std(grads, target_std)
    ]


class _XGBoostEncoder:
    """Per-view ensemble of ``k`` scalar XGBoost boosters, used only during ``fit``.

    Predictions are the raw sum of tree outputs (no base margin / init
    score); the caller is responsible for adding the fixed initial
    embedding.
    """

    def __init__(self, X: np.ndarray, k: int, params: dict[str, object]) -> None:
        self._params = params
        self._dtrain = xgb.DMatrix(X)
        self.boosters: list[xgb.Booster] = [
            xgb.train(params, self._dtrain, num_boost_round=0) for _ in range(k)
        ]

    def predict(self) -> np.ndarray:
        """Raw (base-margin-free) prediction on the training data.

        Returns:
            Array of shape (n_samples, k).
        """
        return np.column_stack(
            [b.predict(self._dtrain, output_margin=True) for b in self.boosters]
        )

    def boost(self, gradient: np.ndarray) -> None:
        """Add one tree to every component booster using the EY gradient.

        Args:
            gradient: EY gradient for this view, shape (n_samples, k).
        """
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
    """Per-view ensemble of ``k`` scalar LightGBM boosters, used only during ``fit``.

    As :class:`_XGBoostEncoder`, but backed by LightGBM's ``Booster`` API.
    """

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
        """Raw (base-margin-free) prediction on the training data.

        Returns:
            Array of shape (n_samples, k).
        """
        return np.column_stack(
            [b.predict(self._X, raw_score=True) for b in self.boosters]
        )

    def boost(self, gradient: np.ndarray) -> None:
        """Add one tree to every component booster using the EY gradient.

        Args:
            gradient: EY gradient for this view, shape (n_samples, k).
        """
        for col, booster in enumerate(self.boosters):
            g = gradient[:, col].copy()

            def _fobj(
                _preds: np.ndarray, _train_set: object, _g: np.ndarray = g
            ) -> tuple[np.ndarray, np.ndarray]:
                return _g, np.ones_like(_g)

            booster.update(fobj=_fobj)


class _CatBoostGradientObjective:
    """Relays one round's fixed target gradient through CatBoost's loss protocol.

    CatBoost's custom-loss objects implement ``calc_ders_range(approxes,
    targets, weights)``, returning per-sample ``(der1, der2)`` -- the
    *negative* first and second derivatives of the loss with respect to the
    current prediction (see CatBoost's own custom-objective examples, e.g.
    ``der1 = target - p`` for log-loss). For the fixed-target squared loss
    used here, that is ``der1 = -gradient`` and the constant
    ``der2 = -1.0`` (a unit Hessian, negated to match CatBoost's sign
    convention) -- the same unit-Hessian Newton step
    :class:`_XGBoostEncoder`/:class:`_LightGBMEncoder` take via their
    ``(gradient, ones_like(gradient))`` objectives. ``targets`` and
    ``weights`` are ignored; the caller supplies a dummy label vector purely
    to satisfy CatBoost's "not all training labels are identical" check.
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
    """Per-view ensemble of ``k`` scalar CatBoost boosters, used only during ``fit``.

    Unlike XGBoost/LightGBM, CatBoost has no notion of an empty, zero-tree
    booster to construct up front, so each component starts as ``None`` (an
    implicit all-zero contribution, handled directly in :meth:`predict`) and
    is replaced wholesale on every :meth:`boost` call by a freshly
    constructed model continued from the previous one via ``init_model``.
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
        """Raw (base-margin-free) prediction on the training data.

        Returns:
            Array of shape (n_samples, k).
        """
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
        """Add one tree to every component booster using the EY gradient.

        Args:
            gradient: EY gradient for this view, shape (n_samples, k).
        """
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
    r"""TreeCCA — nonlinear multiview CCA with gradient-boosted-tree encoders.

    Learns one nonlinear encoder $f_i$ per view (a gradient-boosted
    tree ensemble per latent dimension) that jointly maximise the
    Eckart-Young (EY) unconstrained-CCA objective:

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
    $$

    where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean
    pairwise cross-covariance (including $i = j$ terms) and $V$
    the mean auto-covariance across all views (see
    :mod:`cca_zoo._utils._ey`, the same shared EY-loss machinery used by
    :class:`~cca_zoo.linear.gradient.CCAEY` and
    :class:`~cca_zoo.deep.DCCAEY`). The encoders are fit by alternating
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

    This is an abstract base class shared by every gradient-boosting
    backend: it holds all of the backend-agnostic fitting/transform logic,
    while the choice of tree library lives in a concrete subclass —
    :class:`XGBoostCCA` (the default choice), :class:`LightGBMCCA`, or
    :class:`CatBoostCCA` (the latter two requiring the optional
    ``lightgbm``/``catboost`` packages respectively). Instantiate one of
    those three classes directly; ``TreeCCA`` itself cannot be constructed.

    References:
        Chapman, J. (2026). TreeCCA: Canonical Correlation Analysis via
        Gradient-Boosted Trees. arXiv:2607.27027.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        n_estimators: Number of boosting rounds (trees added per booster).
            Either a single value applied to every view or a list of
            per-view values -- a view whose budget is exhausted first
            stops being boosted (its embedding stays fixed) while the
            others continue. Default is 50.
        max_depth: Maximum depth of each tree. Either a single value
            applied to every view or a list of per-view values. Default
            is 5.
        learning_rate: Boosting learning rate(s). Either a single float
            applied to every view or a list of per-view floats. Default
            is 0.1.
        subsample: Row subsampling ratio(s) per tree. Either a single
            float applied to every view or a list of per-view floats.
            Default is 0.8.
        colsample_bytree: Column subsampling ratio(s) per tree. Either a
            single float applied to every view or a list of per-view
            floats. Default is 0.8.
        min_child_weight: Minimum sum of instance weight (XGBoostCCA) /
            minimum number of samples (LightGBMCCA, CatBoostCCA) needed in
            a child/leaf. Either a single value applied to every view or a
            list of per-view values. Default is 5.
        gauss_seidel: If True, re-predict view 1's embedding after updating
            its boosters and use the fresh values when computing view 2's
            gradient (Gauss-Seidel); if False, both gradients are computed
            from the same stale embeddings (Jacobi). Default is True.
        random_state: Seed for the boosters and for drawing the
            random-orthogonal initial embedding. Default is 0.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        n_estimators: int | list[int] = 50,
        max_depth: int | list[int] = 5,
        learning_rate: float | list[float] = 0.1,
        subsample: float | list[float] = 0.8,
        colsample_bytree: float | list[float] = 0.8,
        min_child_weight: float | list[float] = 5,
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

    @abstractmethod
    def _booster_params(
        self,
        learning_rate: float,
        max_depth: int,
        subsample: float,
        colsample_bytree: float,
        min_child_weight: float,
    ) -> dict[str, object]:
        """Build the backend-specific booster parameter dictionary for one view.

        Args:
            learning_rate: This view's resolved ``learning_rate``.
            max_depth: This view's resolved ``max_depth``.
            subsample: This view's resolved ``subsample``.
            colsample_bytree: This view's resolved ``colsample_bytree``.
            min_child_weight: This view's resolved ``min_child_weight``.

        Returns:
            Dictionary of training parameters for this backend.
        """

    @abstractmethod
    def _make_encoder(
        self, X: np.ndarray, k: int, params: dict[str, object]
    ) -> _XGBoostEncoder | _LightGBMEncoder | _CatBoostEncoder:
        """Construct this backend's per-view encoder.

        Args:
            X: This view's training data, shape (n_samples, n_features).
            k: Number of latent components (boosters in the encoder).
            params: This backend's booster parameters, from
                :meth:`_booster_params`.

        Returns:
            A fresh, zero-tree encoder for this view.
        """

    @abstractmethod
    def _predict_boosters(self, boosters: list[Any], X: np.ndarray) -> np.ndarray:
        """Backend-specific raw prediction for arbitrary (e.g. test) data.

        Args:
            boosters: One fitted booster per latent component.
            X: Input array, shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, k).
        """

    @abstractmethod
    def _importance_example(self) -> str:
        """One-line usage example for per-component feature importance."""

    def fit(self, views: list[ArrayLike], y: None = None) -> TreeCCA:
        """Fit the model.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_ = self._setup_fit(views)
        k = self.latent_dimensions
        n_views = len(views_)

        n_estimators_ = perview_parameter(
            "n_estimators", self.n_estimators, 50, n_views
        )
        max_depth_ = perview_parameter("max_depth", self.max_depth, 5, n_views)
        learning_rate_ = perview_parameter(
            "learning_rate", self.learning_rate, 0.1, n_views
        )
        subsample_ = perview_parameter("subsample", self.subsample, 0.8, n_views)
        colsample_bytree_ = perview_parameter(
            "colsample_bytree", self.colsample_bytree, 0.8, n_views
        )
        min_child_weight_ = perview_parameter(
            "min_child_weight", self.min_child_weight, 5, n_views
        )

        rng = np.random.default_rng(self.random_state)
        base_margins = []
        projections = []
        for X in views_:
            bm, proj = random_orthogonal_embedding(X, k, rng)
            base_margins.append(bm)
            projections.append(proj)
        self._projections_: list[np.ndarray] = projections

        params_per_view = [
            self._booster_params(
                learning_rate_[i],
                max_depth_[i],
                subsample_[i],
                colsample_bytree_[i],
                min_child_weight_[i],
            )
            for i in range(n_views)
        ]
        encoders = [
            self._make_encoder(X, k, p) for X, p in zip(views_, params_per_view)
        ]

        for round_idx in range(max(n_estimators_)):
            representations = [
                bm + enc.predict() for bm, enc in zip(base_margins, encoders)
            ]
            grads = _rescale_to_target_std(ey_grad_z(representations))

            for view_idx in range(n_views):
                if round_idx >= n_estimators_[view_idx]:
                    continue
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
        """Not implemented for TreeCCA models.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: TreeCCA encoders are boosted-tree ensembles,
                not linear weight matrices. Use ``boosters_`` instead.
        """
        check_is_fitted(self)
        raise NotImplementedError(
            f"{type(self).__name__} has no linear weight matrices; its "
            "encoders are gradient-boosted-tree ensembles. Use the "
            "`boosters_` attribute instead for per-component feature "
            f"importance, e.g.\n{self._importance_example()}"
        )


class XGBoostCCA(TreeCCA):
    r"""TreeCCA with XGBoost boosters as the per-view encoders.

    See :class:`TreeCCA` for the shared Eckart-Young objective and
    Gauss-Seidel boosting recipe; this class fixes the gradient-boosting
    backend to `XGBoost <https://xgboost.readthedocs.io/>`_.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 5))
        >>> X2 = rng.standard_normal((100, 5))
        >>> model = XGBoostCCA(latent_dimensions=2, n_estimators=10).fit([X1, X2])
        >>> scores = model.transform([X1, X2])

        A different tree depth and boosting budget per view:

        >>> model = XGBoostCCA(
        ...     latent_dimensions=2, n_estimators=[10, 20], max_depth=[3, 6]
        ... ).fit([X1, X2])
    """

    def _booster_params(
        self,
        learning_rate: float,
        max_depth: int,
        subsample: float,
        colsample_bytree: float,
        min_child_weight: float,
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
            "seed": int(self.random_state),
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

    def _importance_example(self) -> str:
        return 'model.boosters_[view][component].get_score(importance_type="gain")'


class LightGBMCCA(TreeCCA):
    r"""TreeCCA with LightGBM boosters as the per-view encoders.

    See :class:`TreeCCA` for the shared Eckart-Young objective and
    Gauss-Seidel boosting recipe; this class fixes the gradient-boosting
    backend to `LightGBM <https://lightgbm.readthedocs.io/>`_, which
    requires the optional ``lightgbm`` package (``pip install lightgbm``,
    included in the ``tree`` extra).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 5))
        >>> X2 = rng.standard_normal((100, 5))
        >>> model = LightGBMCCA(latent_dimensions=2, n_estimators=10).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    def fit(self, views: list[ArrayLike], y: None = None) -> LightGBMCCA:
        """Fit the model.

        Args: as :meth:`TreeCCA.fit`.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
            ImportError: If the ``lightgbm`` package is not installed.
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
            "min_data_in_bin": 1,
            "verbose": -1,
            "seed": int(self.random_state),
        }

    def _make_encoder(
        self, X: np.ndarray, k: int, params: dict[str, object]
    ) -> _LightGBMEncoder:
        return _LightGBMEncoder(X, k, params)

    def _predict_boosters(self, boosters: list[Any], X: np.ndarray) -> np.ndarray:
        return np.column_stack([b.predict(X, raw_score=True) for b in boosters])

    def _importance_example(self) -> str:
        return (
            "model.boosters_[view][component]"
            '.feature_importance(importance_type="gain")'
        )


class CatBoostCCA(TreeCCA):
    r"""TreeCCA with CatBoost boosters as the per-view encoders.

    See :class:`TreeCCA` for the shared Eckart-Young objective and
    Gauss-Seidel boosting recipe; this class fixes the gradient-boosting
    backend to `CatBoost <https://catboost.ai/>`_, which requires the
    optional ``catboost`` package (``pip install catboost``, included in
    the ``tree`` extra).

    Unlike :class:`XGBoostCCA`/:class:`LightGBMCCA`, which continue an
    existing booster in place, CatBoost has no in-place "add one tree"
    call: each round, every component is replaced by a freshly constructed
    ``CatBoostRegressor(iterations=1, ...)`` continued from the previous
    round's model via ``init_model=``, using a custom loss object
    (:class:`~cca_zoo.tree._treecca._CatBoostGradientObjective`) that
    relays the EY gradient as CatBoost's expected ``(der1, der2)`` pair. As
    a consequence, fitting is markedly slower per round than
    :class:`XGBoostCCA`/:class:`LightGBMCCA` (CatBoost rebuilds its
    training pool and recomputes feature-importance statistics on every
    such call), a cost worth paying when CatBoost's ordered-boosting and
    symmetric-tree structure are themselves the point.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 5))
        >>> X2 = rng.standard_normal((100, 5))
        >>> model = CatBoostCCA(latent_dimensions=2, n_estimators=10).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    def fit(self, views: list[ArrayLike], y: None = None) -> CatBoostCCA:
        """Fit the model.

        Args: as :meth:`TreeCCA.fit`.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
            ImportError: If the ``catboost`` package is not installed.
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
    ) -> dict[str, object]:
        return {
            "depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "bootstrap_type": "Bernoulli",
            "rsm": colsample_bytree,
            "min_data_in_leaf": int(min_child_weight),
            "boost_from_average": False,
            "eval_metric": "RMSE",
            "allow_writing_files": False,
            "verbose": False,
            "random_seed": int(self.random_state),
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

    def _importance_example(self) -> str:
        return "model.boosters_[view][component].get_feature_importance()"
