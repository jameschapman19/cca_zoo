"""GAMCCA — generalized-additive-model Canonical Correlation Analysis."""

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
from cca_zoo._utils._validation import perview_parameter, validate_views

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
    """``mgcv``'s ``m`` for a P-spline: (spline order, penalty order).

    A single value sets both, as in ``mgcv``; the B-splines have degree
    ``order + 1`` (``m=2``: cubic) and the penalty takes ``penalty``-th
    order differences of neighbouring coefficients.
    """
    if isinstance(m, tuple):
        return int(m[0]), int(m[1])
    return int(m), int(m)


class _GamEncoder:
    r"""Per-view additive P-spline encoder: a fixed, centred B-spline basis.

    One smooth per feature, each ``mgcv``'s ``s(x, bs="ps", k=k, m=m)``: a
    ``k``-dimensional B-spline basis on evenly spaced knots, built by
    :class:`~sklearn.preprocessing.SplineTransformer`, with Eilers and
    Marx's difference penalty on neighbouring coefficients, kept as its
    square root (:attr:`penalty_factor_`, the stacked difference matrices,
    unscaled by ``sp``). Every feature keeps its full partition of unity and
    splines with no data under them are kept too — the difference penalty is
    what fills them in. Centring makes each feature's constant coefficient
    vector exactly unidentifiable (a partition of unity centres to zero), so
    it is absorbed structurally, as ``mgcv`` absorbs its sum-to-zero
    constraint: coefficients live in the orthogonal complement of each
    block's constant (:attr:`constraint_`). Any data-dependent rank
    deficiency left (empty spline supports, ties, duplicated features) is
    resolved at fit time by
    :func:`~cca_zoo._utils._ey.full_rank_reparametrisation`. Centring every column
    makes $Z_i = \text{basis}_i B_i$ zero-mean for any coefficients; the
    basis itself stays sparse (B-splines are local), centred only implicitly.
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
        """Encoder output for arbitrary (e.g. test) data, shape (n, k)."""
        result: np.ndarray = self._spline.transform(X) @ self.coef_ - (
            self.basis_mean_ @ self.coef_
        )
        return result

    def feature_term(self, feature_idx: int, x: np.ndarray) -> np.ndarray:
        """Single feature's additive contribution, shape (n, k).

        Args:
            feature_idx: Index of the input feature.
            x: Raw (mean-centred) values for that feature, shape (n,).

        Returns:
            Array of shape (n, k): this feature's term alone, for each
            latent component. Because the basis is centred per *column*
            (not just overall), each feature's own share of that centring
            is exactly ``basis_mean_[block]`` — so summing this over every
            feature reproduces :meth:`predict` exactly, with no leftover
            constant to split arbitrarily across features.
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
    r"""GAMCCA — nonlinear multiview CCA with generalized-additive-model encoders.

    Learns one nonlinear encoder $f_i$ per view — a generalized additive
    model (GAM), $f_i(x) = \sum_j s_{ij}(x_{ij})$, one smooth per input
    feature, each ``mgcv``'s P-spline ``s(x_j, bs="ps", k=k, m=m)`` — that
    jointly minimise the Eckart-Young (EY) objective

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
    $$

    (for embeddings $Z_i = f_i(X_i)$, $C$ the mean pairwise
    cross-covariance including $i = j$ terms and $V$ the mean
    auto-covariance; see :mod:`cca_zoo._utils._ey`), plus each view's
    P-spline smoothing penalty $\tfrac12\,\mathrm{sp}_i \sum_j
    \beta_{ij}^\top D^\top D \beta_{ij}$, $D$ the ``m[1]``-th order
    difference matrix (Eilers and Marx, 1996): the penalty shrinks each
    smooth towards a polynomial of degree ``m[1] - 1`` (a straight line for
    the default ``m=2``), not towards zero.

    With the basis fixed, the whole fit is a single generalized
    eigenproblem (:func:`~cca_zoo._utils._ey.penalised_gram_ey_gep`), solved
    in closed form at its global optimum — the same solver
    :class:`~cca_zoo.gam.MARSCCA` refits with — after an exact
    reparametrisation onto the basis's row space
    (:func:`~cca_zoo._utils._ey.full_rank_reparametrisation`) that resolves
    the P-spline basis's intended rank deficiency, as ``mgcv`` does. There
    is no iterative solve, no convergence tolerance, and no dependence on a
    random start.

    ``mgcv`` estimates each smoothing parameter by GCV or REML; both are
    likelihood/residual criteria with no EY-loss counterpart, so ``sp`` is
    chosen by cross-validation instead, with
    :func:`~cca_zoo.model_selection.one_standard_error` taking the smoothest
    model within one standard error of the best::

        GridSearchCV(
            GAMCCA(),
            {"sp": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
            refit=one_standard_error("sp", larger_is_simpler=True),
        )

    Because each latent component decomposes exactly into one additive term
    per input feature, the fitted shape of any feature's contribution is
    available directly via :meth:`shape_function` — ``plot.gam``'s partial
    effect curves.

    Note:
        A GAM's additive structure assumes each feature contributes
        independently; it cannot represent a genuine *interaction* between
        two features of the same view (e.g. $x_1 x_2$).
        :class:`~cca_zoo.gam.MARSCCA` with ``degree >= 2`` can, as can
        :class:`~cca_zoo.tree.TreeCCA` and
        :class:`~cca_zoo.gp.GaussianProcessCCA`.

    References:
        Wood, S. N. (2017). Generalized Additive Models: An Introduction
        with R (2nd ed.). Chapman and Hall/CRC.

        Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with
        B-splines and penalties. Statistical Science, 11(2), 89-121.

        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        k: Basis dimension of every smooth, as ``mgcv``'s ``k``: the number
            of B-splines per feature. Raise it when a relationship needs more
            wiggles than ten splines allow; the penalty keeps a generous
            basis in check. Cost grows with the cube of the total basis
            size. Either a single value or a list of per-view values.
            Default is 10, ``mgcv``'s.
        m: Spline and penalty orders, as ``mgcv``'s ``m`` for ``bs="ps"``:
            ``(order, penalty_order)`` gives B-splines of degree
            ``order + 1`` and a ``penalty_order``-th difference penalty, and
            a single value sets both. Either one such value (int or tuple)
            or a list of per-view values. Default is 2 (cubic splines,
            second differences), ``mgcv``'s.
        sp: Smoothing parameter of every smooth, as ``mgcv``'s ``sp``
            (penalty ``sp * beta' S beta``); larger is smoother, tending to
            a polynomial of degree ``penalty_order - 1`` per feature. Either
            a single value or a list of per-view values. Default is 0.01
            (with ``k=10``, held-out correlation on smooth nonlinear
            relationships matched or beat ``k=20, sp=0.1`` at a third of the
            cost).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 5))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = GAMCCA(latent_dimensions=2).fit([X1, X2])
        >>> scores = model.transform([X1, X2])

        A different basis size and smoothing per view:

        >>> model = GAMCCA(latent_dimensions=2, k=[8, 12], sp=[0.1, 10.0]).fit(
        ...     [X1, X2]
        ... )
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "k": [Interval(Integral, 4, None, closed="left"), "array-like"],
        "m": [Interval(Integral, 1, None, closed="left"), tuple, "array-like"],
        "sp": [Interval(Real, 0, None, closed="left"), "array-like"],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        k: int | list[int] = 10,
        m: int | tuple[int, int] | list[int | tuple[int, int]] = 2,
        sp: float | list[float] = 0.01,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.k = k
        self.m = m
        self.sp = sp

    def fit(self, views: list[ArrayLike], y: None = None) -> GAMCCA:
        """Fit the GAMCCA model: one closed-form penalised eigenproblem.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
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
            self.latent_dimensions,
            [penalty for _, _, penalty in reduced],
        )
        for enc, (lift_i, _, _), coef in zip(encoders, reduced, coefficients):
            enc.coef_ = enc.constraint_ @ lift_i @ coef
            enc._train_pred = enc.centred_product(enc.coef_)

        self.encoders_: list[_GamEncoder] = encoders
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project views into the latent space using the fitted encoders.

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
        return [enc.predict_new(v) for v, enc in zip(centred, self.encoders_)]

    def shape_function(self, view: int, feature: int, x: ArrayLike) -> np.ndarray:
        r"""Evaluate one feature's fitted additive term $s_j(x_j)$.

        Because GAMCCA's encoder is additive across features, each term can
        be inspected in isolation — the direct GAM analogue of
        :class:`~cca_zoo.tree.TreeCCA`'s split-gain feature importance, but
        an exact, shape-preserving curve rather than a single importance
        score.

        Args:
            view: Index of the view.
            feature: Index of the feature within that view (raw, i.e.
                un-centred column order).
            x: Raw (un-centred) values for that feature at which to evaluate
                the term, shape (n,).

        Returns:
            Array of shape (n, latent_dimensions): that feature's
            contribution alone, for every latent component.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        x_arr = np.asarray(x, dtype=float) - self.means_[view][feature]
        return self.encoders_[view].feature_term(feature, x_arr)

    @property
    def weights(self) -> list[np.ndarray]:
        """Not implemented for GAMCCA.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: GAMCCA encoders are additive splines, not
                linear weight matrices. Use :meth:`shape_function` instead
                to inspect a feature's fitted contribution directly.
        """
        check_is_fitted(self)
        raise NotImplementedError(
            "GAMCCA has no linear weight matrices; its encoders are "
            "generalized additive models (one B-spline term per feature). "
            "Use the `shape_function` method instead to inspect a fitted "
            "feature's contribution directly."
        )
