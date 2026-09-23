r"""Canonical gradient boosting (CanoBoost).

Each boosting round of a Newton gradient-boosting machine fits a tree to the
matrix of per-sample loss gradients ``G`` (n_samples x n_outputs). Standard
implementations (XGBoost, LightGBM, scikit-learn) restrict that tree to
axis-aligned splits, so any signal that lives along an oblique direction has
to be approximated by a staircase of many splits across many rounds.

CanoBoost adds, at every round, the ``r`` directions of feature space that are
maximally canonically correlated with ``G``:

.. math::

    (w_x, w_g) = \arg\max \operatorname{corr}(X w_x, G w_g)

These are exactly the steepest functional-descent directions available to a
linear weak learner: for a single output the leading canonical direction is the
(ridge) least-squares fit of the negative gradient, and for several outputs
(multiclass softmax, multi-target regression) CCA returns the rank-``r``
subspace that explains the most gradient correlation shared across outputs.
The round's tree sees the raw features *and* these canonical variates, so its
hypothesis class strictly contains that of an axis-aligned tree, and one split
on a canonical variate is an oblique split along the direction of steepest
descent.

Because the whitening of ``X`` does not depend on the gradients, it is computed
once; each round then costs one small ``d x K`` SVD on top of the tree fit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit, log_softmax, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import Tags
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from cca_zoo._utils._linalg import svd_whiten


@dataclass
class _Round:
    """One boosting round: projection, tree, and Newton leaf values."""

    projection: np.ndarray  # (n_features, r) acting on standardised X
    tree: DecisionTreeRegressor
    leaf_values: np.ndarray  # (node_count, n_outputs), already scaled by lr


class _BaseCanonicalBoosting(BaseEstimator, ABC):
    """Shared Newton-boosting loop with canonical-correlation feature augmentation.

    Args:
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Shrinkage applied to every tree's leaf values.
        max_depth: Maximum depth of each tree.
        min_samples_leaf: Minimum number of samples in a leaf.
        reg_lambda: L2 penalty on leaf values (Newton denominator ridge).
        n_components: Canonical directions added per round. ``0`` disables the
            augmentation and recovers plain (multi-output) Newton boosting.
        cca_reg: Ridge in ``[0, 1]`` on the feature covariance used to whiten
            ``X``. ``0`` is full CCA (directions are least-squares fits of the
            gradients); ``1`` is PLS (directions maximise covariance).
        subsample: Fraction of rows drawn without replacement for each round.
        random_state: Seed for row subsampling and tree tie-breaking.
    """

    _multi_output: bool

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        n_components: int = 2,
        cca_reg: float = 0.1,
        subsample: float = 0.8,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.n_components = n_components
        self.cca_reg = cca_reg
        self.subsample = subsample
        self.random_state = random_state

    # ------------------------------------------------------------------ loss

    @abstractmethod
    def _fit_targets(self, y: ArrayLike) -> None:
        """Record target metadata (classes, output shape) from training ``y``."""

    @abstractmethod
    def _encode_targets(self, y: ArrayLike) -> np.ndarray:
        """Return targets as an (n_samples, n_outputs) float array."""

    @abstractmethod
    def _init_raw(self, Y: np.ndarray) -> np.ndarray:
        """Constant initial raw prediction, shape (n_outputs,)."""

    @abstractmethod
    def _grad_hess(self, F: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, ...]:
        """Per-sample gradient and diagonal Hessian of the loss at ``F``."""

    @abstractmethod
    def _loss(self, F: np.ndarray, Y: np.ndarray) -> float:
        """Mean loss of raw predictions ``F`` against targets ``Y``."""

    # ------------------------------------------------------------ directions

    def _directions(
        self, Z: np.ndarray, G: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Top canonical directions between whitened features and gradients.

        Args:
            Z: Whitened features for this round's rows, (n, rank_x).
            G: Loss gradients for the same rows, (n, n_outputs).
            rng: Unused here; lets subclasses draw alternative directions.

        Returns:
            Directions in whitened-feature space, (rank_x, r).
        """
        Gw, _ = svd_whiten(G - G.mean(axis=0))
        U, _, _ = np.linalg.svd(Z.T @ Gw, full_matrices=False)
        return U[:, : self.n_components]

    # ------------------------------------------------------------------- fit

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def _augment(self, X: np.ndarray, Xs: np.ndarray, rnd: _Round) -> np.ndarray:
        return np.hstack([X, Xs @ rnd.projection])

    def _raw_update(self, X: np.ndarray, Xs: np.ndarray, rnd: _Round) -> np.ndarray:
        return rnd.leaf_values[rnd.tree.apply(self._augment(X, Xs, rnd))]

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eval_set: tuple[ArrayLike, ArrayLike] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> _BaseCanonicalBoosting:
        """Fit the boosted ensemble.

        Args:
            X: Training features, (n_samples, n_features).
            y: Training targets.
            eval_set: Optional ``(X_val, y_val)`` monitored every round.
            early_stopping_rounds: Stop once the validation loss has not
                improved for this many rounds and keep the best iteration.
                Requires ``eval_set``.

        Returns:
            self
        """
        X, y = validate_data(self, X, y, multi_output=self._multi_output)
        self._fit_targets(y)
        Y = self._encode_targets(y)
        n, _ = X.shape
        rng = np.random.default_rng(self.random_state)

        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        Xs = self._standardise(X)
        Z, self.whitener_ = svd_whiten(Xs, regularization=self.cca_reg)

        self.init_ = self._init_raw(Y)
        F = np.tile(self.init_, (n, 1))

        if eval_set is not None:
            Xv = validate_data(self, eval_set[0], reset=False)
            Yv = self._encode_targets(eval_set[1])
            Xvs = self._standardise(Xv)
            Fv = np.tile(self.init_, (len(Xv), 1))
            self.eval_loss_: list[float] = []
            best, best_iter = np.inf, 0

        self.rounds_: list[_Round] = []
        n_sub = max(1, int(round(self.subsample * n)))
        for it in range(self.n_estimators):
            idx = rng.choice(n, n_sub, replace=False) if n_sub < n else np.arange(n)
            G, H = self._grad_hess(F, Y)
            Gi, Hi = G[idx], H[idx]

            if self.n_components > 0:
                U = self._directions(Z[idx], Gi, rng)
                projection = self.whitener_ @ U
            else:
                projection = np.zeros((X.shape[1], 0))

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(2**31 - 1)),
            )
            A = np.hstack([X[idx], Xs[idx] @ projection])
            tree.fit(A, -Gi if Gi.shape[1] > 1 else -Gi.ravel())

            leaves = tree.apply(A)
            n_nodes = tree.tree_.node_count
            g_sum = np.stack(
                [np.bincount(leaves, Gi[:, k], n_nodes) for k in range(Gi.shape[1])], 1
            )
            h_sum = np.stack(
                [np.bincount(leaves, Hi[:, k], n_nodes) for k in range(Hi.shape[1])], 1
            )
            leaf_values = -self.learning_rate * g_sum / (h_sum + self.reg_lambda)

            rnd = _Round(projection, tree, leaf_values)
            self.rounds_.append(rnd)
            F += self._raw_update(X, Xs, rnd)

            if eval_set is not None:
                Fv += self._raw_update(Xv, Xvs, rnd)
                loss = self._loss(Fv, Yv)
                self.eval_loss_.append(loss)
                if loss < best:
                    best, best_iter = loss, it
                elif (
                    early_stopping_rounds is not None
                    and it - best_iter >= early_stopping_rounds
                ):
                    break

        if eval_set is not None and early_stopping_rounds is not None:
            self.rounds_ = self.rounds_[: best_iter + 1]
        self.best_iteration_ = len(self.rounds_)
        return self

    def _raw_predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "rounds_")
        X = validate_data(self, X, reset=False)
        Xs = self._standardise(X)
        F = np.tile(self.init_, (len(X), 1))
        for rnd in self.rounds_:
            F += self._raw_update(X, Xs, rnd)
        return F

    def canonical_directions(self) -> np.ndarray:
        """Per-round canonical directions in standardised-feature space.

        Returns:
            Array (n_rounds, n_features, r); column ``j`` of round ``t`` is the
            unit-norm feature weighting of that round's ``j``-th canonical
            variate.
        """
        check_is_fitted(self, "rounds_")
        P = np.stack([rnd.projection for rnd in self.rounds_])
        return P / np.linalg.norm(P, axis=1, keepdims=True)


class CanonicalBoostingRegressor(RegressorMixin, _BaseCanonicalBoosting):
    """Canonical gradient-boosted trees for (multi-output) squared-error regression.

    Each round fits a tree on the raw features plus the ``n_components``
    directions most canonically correlated with the current residuals. With
    several targets the directions are shared across outputs, so structure
    common to all targets is found once rather than per target.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.boosting import CanonicalBoostingRegressor
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((300, 5))
        >>> y = np.sin(X @ np.ones(5))
        >>> model = CanonicalBoostingRegressor(
        ...     n_estimators=100, min_samples_leaf=5, random_state=0
        ... )
        >>> model.fit(X, y).score(X, y) > 0.9
        True
    """

    _multi_output = True

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def _fit_targets(self, y: ArrayLike) -> None:
        self._y_ndim = np.ndim(y)

    def _encode_targets(self, y: ArrayLike) -> np.ndarray:
        return np.asarray(y, dtype=float).reshape(len(y), -1)

    def _init_raw(self, Y: np.ndarray) -> np.ndarray:
        return Y.mean(axis=0)

    def _grad_hess(self, F: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, ...]:
        return F - Y, np.ones_like(F)

    def _loss(self, F: np.ndarray, Y: np.ndarray) -> float:
        return float(np.mean((F - Y) ** 2))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict targets.

        Args:
            X: Features, (n_samples, n_features).

        Returns:
            Predictions, (n_samples,) or (n_samples, n_outputs).
        """
        F = self._raw_predict(X)
        return F.ravel() if self._y_ndim == 1 else F


class CanonicalBoostingClassifier(ClassifierMixin, _BaseCanonicalBoosting):
    """Canonical gradient-boosted trees for binary and multiclass classification.

    Binary problems use the logistic loss with one raw score; multiclass
    problems use the softmax loss with one raw score per class and a single
    multi-output tree per round. For multiclass, CCA between the features and
    the (n_samples x n_classes) gradient matrix yields up to ``n_classes - 1``
    shared discriminative directions per round -- a gradient-adaptive
    analogue of Fisher LDA directions, recomputed as the ensemble improves.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from cca_zoo.boosting import CanonicalBoostingClassifier
        >>> X, y = load_iris(return_X_y=True)
        >>> model = CanonicalBoostingClassifier(n_estimators=30, random_state=0)
        >>> model.fit(X, y).score(X, y) > 0.95
        True
    """

    _multi_output = False

    def _fit_targets(self, y: ArrayLike) -> None:
        check_classification_targets(y)
        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        if len(self.classes_) < 2:
            raise ValueError(
                "Classifier needs at least 2 classes; the data has one class: "
                f"{self.classes_}."
            )

    def _encode_targets(self, y: ArrayLike) -> np.ndarray:
        codes = self._label_encoder.transform(y)
        if len(self.classes_) == 2:
            return codes[:, None].astype(float)
        return np.eye(len(self.classes_))[codes]

    def _init_raw(self, Y: np.ndarray) -> np.ndarray:
        p = Y.mean(axis=0)
        return np.log(p / (1 - p)) if Y.shape[1] == 1 else np.log(p)

    def _grad_hess(self, F: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, ...]:
        P = expit(F) if F.shape[1] == 1 else softmax(F, axis=1)
        return P - Y, P * (1 - P)

    def _loss(self, F: np.ndarray, Y: np.ndarray) -> float:
        if F.shape[1] == 1:
            return float(np.mean(np.logaddexp(0, F) - Y * F))
        return float(-np.mean(np.sum(Y * log_softmax(F, axis=1), axis=1)))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Class probabilities.

        Args:
            X: Features, (n_samples, n_features).

        Returns:
            Probabilities, (n_samples, n_classes).
        """
        F = self._raw_predict(X)
        if F.shape[1] == 1:
            p = expit(F[:, 0])
            return np.column_stack([1 - p, p])
        return softmax(F, axis=1)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features, (n_samples, n_features).

        Returns:
            Labels, (n_samples,).
        """
        P = self.predict_proba(X)
        return self.classes_[np.argmax(P, axis=1)]
