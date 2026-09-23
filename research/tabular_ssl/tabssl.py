"""Non-deep self-supervised representations for tabular data.

All methods work in *copula space*: each column is mapped through its
empirical CDF and then the standard-normal quantile function. This enforces
the one symmetry every tabular column has -- invariance to monotone
reparametrisation (units, log transforms, rank-preserving recodings) -- and
makes the joint approximately Gaussian so conditionals have closed forms.

Level 1 (``SpectralSSL``): a linear encoder trained with a LeJEPA-style
objective -- make two augmented views of a row agree, subject to whitened
(non-collapsed) embeddings. With a linear encoder and Gaussian data that
objective has an exact solution: a generalised eigenproblem

    C_cross w = lambda C_view w,

where ``C_cross`` is the covariance between two independent augmentations of
the same row and ``C_view`` the covariance of one augmented row. ``lambda``
is the expected view agreement of a component, so it is a direct, label-free
measure of how much of that direction survives the augmentation.

Two augmentations are provided:

* ``"gibbs"`` -- resample a random subset of columns from their conditional
  distribution given the other columns (a blocked Gibbs step under the
  Gaussian copula). It preserves the joint distribution, and it perturbs a
  column in proportion to how *unpredictable* it is from the rest: redundant
  columns barely move, idiosyncratic ones are replaced.
* ``"marginal"`` -- resample the chosen columns from their marginals (the
  SCARF corruption). It destroys the dependence between the resampled and
  kept columns and produces off-manifold rows.

Level 2 (``ConditionalSSL``): the pretext task is the downstream task family
itself -- predict every column from all the others -- solved with gradient-
boosted trees, which already carry the tabular inductive bias. Its output is
each column's cross-fitted prediction, i.e. the part of the column that is
shared with (redundantly encoded by) the rest of the table.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from scipy.linalg import eigh
from scipy.special import ndtri
from sklearn.model_selection import KFold


class CopulaTransform:
    """Per-column empirical-CDF -> standard-normal-quantile transform."""

    def fit(self, X: np.ndarray) -> CopulaTransform:
        """Store the sorted training values of every column."""
        self.sorted_ = [np.sort(col) for col in X.T]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map to copula space using mid-ranks against the training values."""
        Z = np.empty_like(X, dtype=float)
        for j, ref in enumerate(self.sorted_):
            lo = np.searchsorted(ref, X[:, j], side="left")
            hi = np.searchsorted(ref, X[:, j], side="right")
            Z[:, j] = ndtri((0.5 * (lo + hi) + 0.5) / (len(ref) + 1))
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X).transform(X)


def _covariance(Z: np.ndarray, shrinkage: float) -> np.ndarray:
    S = np.cov(Z, rowvar=False)
    return (1 - shrinkage) * S + shrinkage * np.diag(np.diag(S))


class SpectralSSL:
    """Closed-form linear view-agreement SSL under the Gaussian copula.

    Args:
        augmentation: ``"gibbs"`` (conditional resampling) or ``"marginal"``
            (SCARF-style resampling from the column marginals).
        mask_rate: Probability that each column is resampled in a view.
        n_masks: Monte Carlo draws of the column mask.
        shrinkage: Shrinkage of the copula covariance towards its diagonal.
        random_state: Seed for the mask draws.
    """

    def __init__(
        self,
        augmentation: str = "gibbs",
        mask_rate: float = 0.5,
        n_masks: int = 256,
        shrinkage: float = 0.05,
        random_state: int = 0,
    ) -> None:
        self.augmentation = augmentation
        self.mask_rate = mask_rate
        self.n_masks = n_masks
        self.shrinkage = shrinkage
        self.random_state = random_state

    def fit(self, Z: np.ndarray) -> SpectralSSL:
        """Solve for the encoder on copula-space data ``Z``."""
        d = Z.shape[1]
        S = _covariance(Z, self.shrinkage)
        rng = np.random.default_rng(self.random_state)
        B_mean = np.zeros((d, d))
        C_view = np.zeros((d, d))
        for _ in range(self.n_masks):
            m = rng.random(d) < self.mask_rate
            if m.all():
                m[rng.integers(d)] = False
            o = ~m
            # E[view | row] = B @ row: kept columns copied, masked ones set to
            # their conditional mean (gibbs) or marginal mean 0 (marginal).
            B = np.eye(d)
            B[m] = 0.0
            if self.augmentation == "gibbs":
                B[np.ix_(m, o)] = np.linalg.solve(S[np.ix_(o, o)], S[np.ix_(o, m)]).T
                C_view += S  # conditional resampling preserves the joint
            else:
                C_view += B @ S @ B.T + np.diag(m.astype(float))
            B_mean += B
        B_mean /= self.n_masks
        C_view /= self.n_masks
        # Two views draw independent masks and noise, so their cross-covariance
        # factorises through the mean map.
        C_cross = B_mean @ S @ B_mean.T
        lam, W = eigh(C_cross, C_view)
        order = np.argsort(lam)[::-1]
        self.agreement_ = lam[order]
        self.components_ = W[:, order]
        return self

    def transform(self, Z: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Project onto the top ``n_components`` agreement directions."""
        return Z @ self.components_[:, :n_components]


class ConditionalSSL:
    """Predict every column from the others with cross-fitted boosted trees.

    Args:
        n_folds: Cross-fitting folds; training rows get out-of-fold
            predictions so they are distributed like predictions on new rows.
        n_estimators: Trees per column model.
        learning_rate: LightGBM learning rate.
        num_leaves: LightGBM leaves per tree.
        random_state: Seed for folds and LightGBM.
    """

    def __init__(
        self,
        n_folds: int = 5,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        random_state: int = 0,
    ) -> None:
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state

    def _model(self) -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1,
        )

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        """Fit per-column models; return out-of-fold predictions for ``Z``."""
        n, d = Z.shape
        folds = list(
            KFold(self.n_folds, shuffle=True, random_state=self.random_state).split(Z)
        )
        self.models_: list[list[LGBMRegressor]] = []
        Zhat = np.empty_like(Z)
        for j in range(d):
            rest = np.delete(np.arange(d), j)
            fold_models = []
            for tr, te in folds:
                m = self._model().fit(Z[tr][:, rest], Z[tr, j])
                Zhat[te, j] = m.predict(Z[te][:, rest])
                fold_models.append(m)
            self.models_.append(fold_models)
        return Zhat

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Predict each column of new rows, averaging the fold models."""
        d = Z.shape[1]
        Zhat = np.empty_like(Z)
        for j, fold_models in enumerate(self.models_):
            rest = np.delete(np.arange(d), j)
            Zhat[:, j] = np.mean([m.predict(Z[:, rest]) for m in fold_models], axis=0)
        return Zhat
