"""Generalized additive model CCA."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from sklearn.preprocessing import SplineTransformer
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    full_rank_reparametrisation,
    penalised_gram_ey_closed_form,
)
from cca_zoo._utils._validation import perview_parameter

# Rows of the sparse basis densified at a time to form its Gram with BLAS:
# ~8x faster than a sparse-sparse product (whose output is dense anyway,
# every pair of features overlapping), at a bounded 1024 x d of memory.
_GRAM_CHUNK = 1024


def _sparse_gram(basis: sparse.csr_array) -> np.ndarray:
    """``basis.T @ basis`` as a dense array, by BLAS over row chunks."""
    gram = np.zeros((basis.shape[1], basis.shape[1]))
    for start in range(0, basis.shape[0], _GRAM_CHUNK):
        chunk = basis[start : start + _GRAM_CHUNK].toarray()
        gram += chunk.T @ chunk
    return gram


def _spline_orders(m: int | tuple[int, int]) -> tuple[int, int]:
    """``mgcv``'s P-spline ``m`` as ``(order, penalty_order)``; an int sets both."""
    if isinstance(m, tuple):
        return int(m[0]), int(m[1])
    return int(m), int(m)


class _GamEncoder:
    """Per-view additive P-spline encoder on a fixed B-spline basis.

    One ``mgcv`` smooth ``s(x, bs="ps", k=k, m=m)`` per feature, built with
    :class:`~sklearn.preprocessing.SplineTransformer`. The difference penalty
    is kept as its square root, ``penalty_factor_``. Coefficients are
    constrained to the complement of each feature's constant
    (``constraint_``), which centring makes unidentifiable, as ``mgcv``'s
    sum-to-zero constraint does. The basis stays sparse and is centred
    implicitly.
    """

    def __init__(self, X: np.ndarray, k: int, m: int | tuple[int, int]) -> None:
        self.n, self.p = X.shape
        order, penalty_order = _spline_orders(m)
        self._spline = SplineTransformer(
            n_knots=k - order,
            degree=order + 1,
            knots="uniform",
            extrapolation="linear",
            include_bias=True,
            sparse_output=True,
        )
        # B-splines are local: each row has order + 2 nonzeros per feature, so
        # the basis is kept sparse and centred only implicitly (its mean is
        # subtracted wherever a product with it is taken).
        self.raw_basis_: sparse.csr_array = sparse.csr_array(
            self._spline.fit_transform(X)
        )
        self.n_splines_: int = self.raw_basis_.shape[1] // self.p
        self.basis_mean_: np.ndarray = np.asarray(self.raw_basis_.mean(axis=0)).ravel()
        differences = np.diff(np.eye(self.n_splines_), n=penalty_order, axis=0)
        # The penalty's square root, F with R = F.T @ F: the difference
        # matrix of every feature's coefficients.
        self.penalty_factor_: np.ndarray = scipy.linalg.block_diag(
            *([differences] * self.p)
        )
        self.penalty_norm_: float = float(np.linalg.norm(differences, 2))
        # Each block's constant is absorbed as mgcv absorbs its constraint:
        # a Householder reflection H = I - 2uu' maps it onto the block's first
        # axis, and H's other columns span its orthogonal complement. The
        # difference penalty annihilates constants, so this changes neither
        # fit nor penalty — it only removes a direction the data cannot
        # identify.
        reflector = np.ones(self.n_splines_)
        reflector[0] += np.sqrt(self.n_splines_)
        reflector /= np.linalg.norm(reflector)
        self.reflectors_: np.ndarray = scipy.linalg.block_diag(
            *([reflector[:, None]] * self.p)
        )
        self.free_: np.ndarray = (
            np.arange(self.p * self.n_splines_) % self.n_splines_ > 0
        )
        self.constraint_: np.ndarray = (
            np.eye(self.p * self.n_splines_) - 2 * self.reflectors_ @ self.reflectors_.T
        )[:, self.free_]
        self.coef_: np.ndarray = np.zeros((self.raw_basis_.shape[1], 1))
        self._train_pred: np.ndarray = np.zeros((self.n, 1))

    def centred_product(self, coef: np.ndarray) -> np.ndarray:
        """The centred training basis times ``coef``, without densifying it."""
        result: np.ndarray = self.raw_basis_ @ coef - self.basis_mean_ @ coef
        return result

    def basis_operator(self, scale: float) -> LinearOperator:
        """The centred, constrained training basis over ``scale``, unformed."""

        def product(coef: np.ndarray) -> np.ndarray:
            return self.centred_product(self.constraint_ @ coef) / scale

        shape = (self.n, self.constraint_.shape[1])
        return LinearOperator(shape, matvec=product, matmat=product)

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        return self._train_pred

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """Encoder output for new data, shape (n, k)."""
        result: np.ndarray = self._spline.transform(X) @ self.coef_ - (
            self.basis_mean_ @ self.coef_
        )
        return result

    def feature_term(self, feature_idx: int, x: np.ndarray) -> np.ndarray:
        """One feature's additive term, shape (n, k); the terms sum to :meth:`predict`.

        Args:
            feature_idx: Index of the feature.
            x: Mean-centred values of that feature, shape (n,).
        """
        grid = np.zeros((len(x), self.p))
        grid[:, feature_idx] = x
        raw_basis = sparse.csr_array(self._spline.transform(grid))
        block = slice(
            feature_idx * self.n_splines_, (feature_idx + 1) * self.n_splines_
        )
        coef = self.coef_[block]
        result: np.ndarray = raw_basis[:, block] @ coef - self.basis_mean_[block] @ coef
        return result


class GAMCCA(BaseModel):
    r"""Nonlinear CCA with generalized additive model encoders.

    Each view's encoder is $f_i(x) = \sum_j s_{ij}(x_j)$, one ``mgcv``
    P-spline smooth ``s(x_j, bs="ps", k=k, m=m)`` per feature, fitted to
    minimise the EY loss (:mod:`cca_zoo._utils._ey`) plus the smoothing
    penalty $\tfrac12 \mathrm{sp}_i \sum_j \beta_{ij}^\top D^\top D \beta_{ij}$,
    with $D$ the ``m[1]``-th order difference matrix. The fit is a single
    generalized eigenproblem, solved in closed form at its global optimum.
    ``mgcv``'s GCV and REML have no EY counterpart, so choose ``sp`` by
    cross-validation. An additive model cannot represent within-view
    interactions; see :class:`~cca_zoo.gam.MARSCCA`.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        k: B-splines per feature, as ``mgcv``'s ``k``. Per-view. Default is 10.
        m: ``(order, penalty_order)``, as ``mgcv``'s ``m``: splines of degree
            ``order + 1`` and a ``penalty_order``-th difference penalty; an
            int sets both. Per-view. Default is 2.
        sp: Smoothing parameter; larger is smoother. Per-view. Default is 0.01.

    Attributes:
        encoders_: Fitted per-view encoders.

    References:
        Wood, S. N. (2017). Generalized Additive Models: An Introduction
        with R (2nd ed.). Chapman and Hall/CRC.

        Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with
        B-splines and penalties. Statistical Science, 11(2), 89-121.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.gam import GAMCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 5))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = GAMCCA(n_components=2, k=[8, 12], sp=[0.1, 10.0]).fit([X1, X2])
        >>> model.shape_function(view=0, feature=0, x=X1[:, 0]).shape
        (200, 2)
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "k": [Interval(Integral, 4, None, closed="left"), "array-like"],
        "m": [Interval(Integral, 1, None, closed="left"), tuple, "array-like"],
        "sp": [Interval(Real, 0, None, closed="left"), "array-like"],
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        k: int | list[int] = 10,
        m: int | tuple[int, int] | list[int | tuple[int, int]] = 2,
        sp: float | list[float] = 0.01,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.k = k
        self.m = m
        self.sp = sp

    def fit(self, views: list[ArrayLike], y: None = None) -> GAMCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ValueError: If ``k`` is too small for a view's spline order.
        """
        views_ = self._setup_fit(views)
        m_views = self.n_views_
        k_ = perview_parameter("k", self.k, 10, m_views)
        m_: list[int | tuple[int, int]] = perview_parameter("m", self.m, 2, m_views)
        sp_ = perview_parameter("sp", self.sp, 0.01, m_views)
        for i, (k, m) in enumerate(zip(k_, m_)):
            order, penalty_order = _spline_orders(m)
            if k < order + 2 or penalty_order >= k:
                raise ValueError(
                    f"k={k} is too small for m={m} in view {i}: a P-spline of "
                    f"order {order} needs k >= {order + 2}, and the penalty "
                    f"order must be below k."
                )
        encoders = [_GamEncoder(X, k, m) for X, k, m in zip(views_, k_, m_)]
        # The stacked Gram of the centred, constrained bases, from the sparse
        # raw bases: Z'(S - 1 mu')'(S - 1 mu')Z = Z'(S'S - n mu mu')Z.
        stacked = sparse.hstack([enc.raw_basis_ for enc in encoders]).tocsr()
        mean = np.concatenate([enc.basis_mean_ for enc in encoders])
        n = stacked.shape[0]
        scale = m_views * (n - 1)
        raw_gram = _sparse_gram(stacked) - n * np.outer(mean, mean)
        # H G H for the block-diagonal reflection H = I - 2UU' is a rank-2p
        # update of G, O(d^2 p) rather than two dense d^3 products.
        reflectors = scipy.linalg.block_diag(*[enc.reflectors_ for enc in encoders])
        gram_reflectors = raw_gram @ reflectors
        reflected = (
            raw_gram
            - 2 * reflectors @ gram_reflectors.T
            - 2 * gram_reflectors @ reflectors.T
            + 4 * reflectors @ (reflectors.T @ gram_reflectors) @ reflectors.T
        )
        free = np.concatenate([enc.free_ for enc in encoders])
        gram = reflected[np.ix_(free, free)] / scale
        view = np.repeat(
            np.arange(m_views), [enc.constraint_.shape[1] for enc in encoders]
        )
        reduced = [
            full_rank_reparametrisation(
                enc.basis_operator(np.sqrt(scale)),
                gram[np.ix_(view == i, view == i)],
                np.sqrt(sp) * enc.penalty_factor_ @ enc.constraint_,
                np.sqrt(sp) * enc.penalty_norm_,
            )
            for i, (enc, sp) in enumerate(zip(encoders, sp_))
        ]
        rows = [row_i for _, row_i, _ in reduced]
        reduced_gram = np.block(
            [
                [
                    rows[i].T @ gram[np.ix_(view == i, view == j)] @ rows[j]
                    for j in range(m_views)
                ]
                for i in range(m_views)
            ]
        )
        reduced_view = np.repeat(np.arange(m_views), [r.shape[1] for r in rows])
        coefficients = penalised_gram_ey_closed_form(
            reduced_gram,
            reduced_view,
            self.n_components,
            [penalty for _, _, penalty in reduced],
        )
        for enc, (lift_i, _, _), coef in zip(encoders, reduced, coefficients):
            enc.coef_ = enc.constraint_ @ lift_i @ coef
            enc._train_pred = enc.centred_product(enc.coef_)

        self.encoders_: list[_GamEncoder] = encoders
        return self

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        return self.encoders_[view].predict_new(centred)

    def _feature_importances(self) -> list[np.ndarray]:
        """Variance of each feature's smooth over the training data."""
        return [
            np.array(
                [
                    enc.feature_term(j, train[:, j]).var(axis=0).sum()
                    for j in range(train.shape[1])
                ]
            )
            for enc, train in zip(self.encoders_, self._views_fit_)
        ]

    def shape_function(self, view: int, feature: int, x: ArrayLike) -> np.ndarray:
        """Evaluate one feature's fitted smooth, ``plot.gam``'s partial effect.

        Args:
            view: Index of the view.
            feature: Index of the feature.
            x: Raw values of that feature, shape (n,).

        Returns:
            The feature's contribution, shape (n, n_components).
        """
        check_is_fitted(self)
        x_arr = np.asarray(x, dtype=float) - self.means_[view][feature]
        return self.encoders_[view].feature_term(feature, x_arr)
