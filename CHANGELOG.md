# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `MARSCCA` (in `cca_zoo.gam`): nonlinear multiview CCA using a multivariate adaptive
  regression spline (Friedman, 1991) as the per-view encoder, trained on the same
  Eckart-Young objective as `GAMCCA`. Where `GAMCCA` fixes a B-spline basis up front,
  `MARSCCA` grows each view's basis greedily as classical MARS does — every forward step
  adds the reflected hinge pair (any existing term as parent, any feature, any knot
  `earth`'s `minspan`/`endspan` rules allow within the parent's support) that absorbs the
  most of the current EY gradient, with the top ten re-ranked by their exact refit
  loss (`earth`'s criterion), then refits every view's coefficients jointly, stopping
  early by `earth`'s `thresh` rule — so knots go only where cross-view signal needs them, and
  `degree >= 2` admits within-view interactions that an additive model cannot
  represent. As in `earth`, a backward pass (`nprune`) then deletes, one at a time, the
  term whose removal raises the refit training EY loss least; the size is chosen by
  searching `nprune` with `GridSearchCV`, `earth`'s `pmethod="cv"` (its default, GCV, has
  no EY-loss counterpart). Every refit is the
  closed-form optimum of a generalized eigenproblem, and deleting a term restricts it by
  one linear constraint, so each backward step scores every candidate exactly from one
  eigendecomposition (a secular-equation count via Sylvester's law of inertia). Selected
  terms are inspectable via `model.basis_functions(view)`, and its
  `feature_importances_` is `earth`'s `evimp` (the loss criterion).
  Parameters take `earth`'s names and defaults (`degree`, `nk`, `nprune`, `thresh`,
  `minspan`, `endspan`), so an `earth` user can read a call directly. The one default
  that differs is `minspan`: `minspan=0` is Friedman's spacing exactly, as in `earth`,
  while the default widens it to at most 20 knots per feature, which pruned held-out
  correlation on a pure three-way interaction favours 0.94 to 0.61. Candidate scoring uses Friedman's
  suffix-sum fast update, evaluated for every parent, feature and knot at once by
  sparse block-membership matrices built once per fit. Each candidate's projection onto
  the (incrementally orthonormalised) basis is cached as three running scalars and the
  gradient is projected off the basis once per step, so no candidate column is ever
  formed and memory stays O(n_samples * n_features) regardless of `nk` or
  `degree`.

- `feature_importances_` on every model: one non-negative array per view, summing to 1,
  computed on access as for sklearn's tree models. Each family uses its own literature's
  importance — a linear model's `Var(x_j) * sum_k w_jk**2`, `GAMCCA` each smooth's variance,
  `MARSCCA` `earth`'s `evimp`, the tree models their total split gain — and models with no
  such decomposition (kernel, Gaussian-process, manifold) the mean squared change in a
  view's latent scores when a feature is permuted, which for a linear or additive model is
  exactly twice its variance share.

- `ProbabilisticCCA`, `VariationalBayesCCA` and `GFA` gain `posterior_mean(views)`, the
  posterior mean of the shared latent given every view or, with `None` entries, any subset.

- `GridSearchCV` and `RandomizedSearchCV` accept a callable `refit`, as sklearn's do, and
  hand it `cv_results_` with the same unprefixed parameter names as their own
  `cv_results_`. The model-selection guide shows the one-standard-error rule written this way.

- `cca_zoo._utils._ey.penalised_basis_ey_gep` / `penalised_basis_ey_closed_form`: the
  ridge-penalised EY fit on fixed bases is the generalized eigenproblem
  `(A - R/4) W = B W (WᵀBW)`, solved in closed form at its global optimum (loss
  `-Σ μ²` over the top-k eigenvalues), used by `MARSCCA` for every refit.

### Changed

- `GAMCCA` now follows `mgcv`'s API and P-spline smooths. This is a breaking change with
  no deprecation shim, as was `TreeCCA`'s `backend=` removal, since `GAMCCA` shipped only
  in 3.3.0. Each feature's smooth is `s(x, bs="ps", k=k, m=m)`: `k` B-splines on evenly
  spaced knots with Eilers and Marx's difference penalty, which shrinks towards a
  polynomial — the default towards a straight line — rather than the ridge towards zero
  the previous version applied despite citing P-splines. `n_knots` and `alpha` become `k`
  (default 10, `mgcv`'s) and `sp` (default 0.01), and
  `m` is new. The fit is now one closed-form generalized eigenproblem at the global
  optimum, so `max_iter`, `tol` and `random_state` are removed. Each feature's constant, which centring makes
  unidentifiable, is absorbed by a Householder reflection as `mgcv` absorbs its
  sum-to-zero constraint; any data-dependent rank deficiency left (data-free splines,
  ties, duplicated features) is resolved by an exact reparametrisation onto the row space
  (`cca_zoo._utils._ey.full_rank_reparametrisation`). The basis stays sparse, and
  everything after its Gram is $d \times d$: a fit with 100 features and 5000 samples
  takes about half a second. Held-out
  correlation over quadratic, sine, absolute-value and linear relationships rises from
  0.51 to 0.61 on average. `sp` is chosen by cross-validation, since `mgcv`'s GCV/REML
  have no EY-loss counterpart.
- `cca_zoo._utils._ey`: the ridge-only fixed-basis solver becomes
  `penalised_basis_ey_closed_form` / `penalised_basis_ey_gep` / `penalised_basis_ey_min_loss`,
  taking any quadratic penalty per view (a ridge or a matrix); the trust-region solver it
  replaced is removed. Each has a Gram-level counterpart (`penalised_gram_ey_gep`,
  `penalised_gram_ey_closed_form`) for callers that form the Gram themselves, as
  `GAMCCA` does from its sparse basis.

- **Breaking:** `score` returns one float, the mean canonical correlation, as sklearn's
  contract for `score` requires; it was an array of per-dimension correlations. Those
  come from `cca_zoo.metrics`:
  `average_pairwise_correlations(pairwise_correlations(model.transform(views)))`.
- **Breaking:** the probabilistic models' `transform` returns one projection `x_i @ W_i`
  per view, like every other model, instead of a single-element list holding the joint
  posterior mean (now `posterior_mean`). Their special-cased `score`, correlation and
  loading methods are gone, since the shared ones now apply.
- `transform`, `predict` and `inverse_transform` all go through one per-view encoder,
  `BaseModel._transform_view`: a linear model's projection onto `weights_`, overridden
  by each nonlinear model. `predict` estimates the shared latent from the observed views
  through it (the posterior mean, for the probabilistic models).
- `ManifoldCCA`'s training embedding is `embedding_`, the name sklearn's manifold learners
  use, rather than `weights_`, which elsewhere means weight matrices.

### Deprecated

- The `weights` property: use the `weights_` attribute.
- `pairwise_correlations`, `average_pairwise_correlations` and `get_factor_loadings` as
  model methods: the functions of the same names in `cca_zoo.metrics` take a model's
  `transform` output.
- `cca_zoo.model_selection.procrustes_rotation`: it is
  `scipy.linalg.orthogonal_procrustes`, which the permutation test now calls directly.
- `ManifoldCCA.weights_`: use `embedding_`.

### Fixed

- `predict` and `inverse_transform` projected with `weights_` directly, so they raised or
  returned the wrong shape for every nonlinear model (`GAMCCA`, `MARSCCA`,
  `GaussianProcessCCA`, the kernel, manifold and tree models); they now use each model's
  own encoder.
- `predict` and `inverse_transform` cached their reconstruction loadings on first use and
  never invalidated them, so after refitting on new data they silently used the old
  data's; they are now computed on demand.
- `KCCA`, `KGCCA` and `KTCCA` formed test kernels from uncentred inputs against centred
  training data, so `transform` of data far from the origin collapsed (an RBF kernel to
  zero); inputs are now centred as in `fit`.

### Performance

Same results (to rounding, or up to a rotation inside a degenerate eigenspace), less time:

- `KTCCA` and `TCCA` form the cross-moment tensor by one BLAS contraction
  (`cca_zoo._utils._linalg.cross_moment_tensor`) instead of materialising every
  sample's outer product before averaging — for two kernel views an n x n x n
  intermediate (8 GB at n = 1000) for what is `W1ᵀW2 / n` — and whiten with one
  symmetric `eigh` (`psd_inverse_sqrt`) instead of `inv(sqrtm(·))`. `KTCCA` at n = 1000
  drops from 49 s to under 1 s; `TCCA` about 19x.
- `GCCA` takes its shared latent space from the thin SVD of the stacked whitened views
  rather than an eigendecomposition of the n x n matrix they span: O(n p²) instead of
  O(n³), 17x at n = 4000.
- `SAR` passes its lasso paths the Gram matrix and skips sklearn's per-call input
  validation (the documented fast path, used exactly where `precompute="auto"` would
  build the Gram): 1.7x, identical coefficients.
- `ProjectionPursuitCCA`'s Spearman index ranks with `scipy.stats.rankdata` directly
  instead of through `spearmanr`'s per-call overhead, and its angle parametrisation is
  one cumulative product: 1.8x.

## [3.3.0] - 2026-09-22

### Added

- Per-view hyperparameters for every nonlinear-encoder model that was previously
  restricted to a single value shared across all views: `TreeCCA` (and its
  `XGBoostCCA`/`LightGBMCCA`/`CatBoostCCA` backends -- `n_estimators`, `max_depth`,
  `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`),
  `GaussianProcessCCA` (`kernel`, `alpha`, `n_inducing`), `GAMCCA` (`n_knots`,
  `alpha`), `ManifoldCCA` (`n_neighbors`, `affinity`, `gamma`, `lle_reg`,
  `n_operator_components` -- `method` itself stays global; see the class
  docstring's `Note` for why), `ElasticNetCCA` (`alpha`, `l1_ratio`), and
  `MultiTaskElasticNetCCA` (`alpha`, `l1_ratio`). Each now accepts either a single
  value applied to every view (unchanged default behaviour) or a list with one
  value per view, via the same `perview_parameter` scalar-or-list convention
  already used by `MCCA`/`rCCA`/`KCCA`/`GCCA`/`TCCA`/`GraphicalLassoCCA`/etc. --
  sharing a single value across every view is the narrower case, not the default
  one, and every model whose math actually supports varying it independently
  per view should expose that rather than silently forcing it to be shared. For
  `TreeCCA`, a view whose `n_estimators` budget is exhausted first simply stops
  being boosted (its embedding stays fixed) while the other views continue.
- `cca_zoo.metrics`: a new module of plain functions (operating on already-computed
  arrays, the same convention `sklearn.metrics` uses -- not on a fitted model or raw
  views) for evaluating fitted multiview CCA models. `pairwise_correlations`,
  `average_pairwise_correlations` and `factor_loadings` are the exact math
  `BaseModel`'s own methods of the same name were computing inline -- extracted here
  so it's written, and tested, exactly once, and reused by both `BaseModel` and the
  probabilistic module's `PosteriorMeanTransformMixin` (which previously duplicated
  all three almost verbatim, differing only in which per-view projection fed them).
  Also adds two literature metrics that didn't exist anywhere in the package before:
  `adequacy_coefficient` (Cramer & Nicewander, 1979 -- a.k.a. per-dimension
  communality: the proportion of a view's own variance its canonical variates
  capture) and `redundancy_index`/`total_redundancy` (Stewart & Love, 1968 -- the
  proportion of one view's own variance explained *via* another view's canonical
  variate, built compositionally from `adequacy_coefficient` and
  `pairwise_correlations` rather than re-deriving anything). Unlike a canonical
  correlation, redundancy is asymmetric between views and answers a different
  question: how useful a canonical variate actually is for reconstructing a view's
  own features, not just how correlated the views' variates are with each other.
- `cca_zoo.model_selection.HalvingGridSearchCV`/`HalvingRandomSearchCV`: multiview
  adapters around `sklearn.model_selection.HalvingGridSearchCV`/`HalvingRandomSearchCV`,
  following the same `MultiviewWrapper` pattern as `GridSearchCV`/`RandomizedSearchCV` --
  most candidates are eliminated early on a small subset of the training samples, and
  only the survivors are evaluated on progressively larger subsets, which is usually
  much cheaper than an exhaustive search over a large grid. The "resource" grown between
  rounds is a row count of the already view-concatenated training array, so the
  successive-halving mechanics need no multiview-specific handling. Both classes now
  share their `fit` body with `GridSearchCV`/`RandomizedSearchCV` via a new
  `_BaseMultiviewSearchCV` base.
- `cca_zoo.preprocessing.PerViewTransformer`: applies an sklearn transformer (e.g.
  `StandardScaler`, `SimpleImputer`, `PCA`, `KernelCenterer`) independently to each view --
  a fresh clone is fit per view, so fitted state (a scaler's mean, an imputer's fill
  value, ...) is never shared across views -- or a list of one transformer per view for
  heterogeneous preprocessing. Because it preserves the `list[ArrayLike]` views
  convention on both `fit` and `transform`, it composes directly with
  `sklearn.pipeline.Pipeline` (and, through that, `cca_zoo.model_selection.GridSearchCV`/
  `RandomizedSearchCV`) with a cca_zoo multiview estimator as the final step: no
  dedicated multiview `Pipeline` class was needed.
- `ECCA`: reduced-rank-regression CCA with an entrywise L1 penalty, the companion to
  `CCAR3`'s row-group-lasso penalty -- a feature can now contribute to one canonical
  component while being dropped from another, rather than being all-or-nothing across
  every component the way `CCAR3`'s row-group penalty forces. A NumPy port of the
  `ccar3` R package's `ecca()`. Because an entrywise penalty places no coupling between
  a coefficient row's entries, the underlying convex problem separates exactly into one
  independent Lasso regression per response column, so it's fit by a bank of
  `sklearn.linear_model.Lasso` fits rather than the R package's single matrix-free ADMM
  over the whole coefficient matrix (needed there for large-`p`-and-`q` memory
  efficiency, unnecessary here since sklearn's own coordinate descent already handles
  `p > n` natively per column) -- verified to reach the same optimum as an independent
  proximal-gradient (ISTA) solve of the joint objective to high precision. Unlike
  `CCAR3`, deliberately does *not* Ledoit-Wolf whiten `Y` before fitting -- the R
  reference's own `ecca()` ignores its `Sy` argument entirely, regressing directly
  against raw (centred) `Y` -- verified against the R reference directly (installed R,
  sourced `ecca()`'s ADMM): matching support size and canonical correlations at nearby
  `lambda_`. Shares `CCAR3`'s postprocessing machinery, now factored into
  `cca_zoo.linear._rrr_common` along with `CCAR3`'s own Y-whitening helper.
- `ManifoldCCA`: transductive multiview CCA where each view's within-view constraint is
  a graph operator from spectral manifold learning -- the graph Laplacian (matching
  `sklearn.manifold.SpectralEmbedding`, `method="laplacian"`) or the LLE local
  reconstruction operator (`method="lle"`, hand-implemented -- `LocallyLinearEmbedding`
  doesn't expose it as public API) -- in place of a covariance matrix. Solves the joint,
  multiview generalised eigenproblem this induces directly (each view's "weight" *is*
  its training-set embedding, since there's no feature map, only a per-view graph built
  from that view's own local neighbourhood structure), then extends out of sample via
  each method's own established mechanism rather than a generic auxiliary model:
  `method="lle"` reuses `LocallyLinearEmbedding.transform`'s own barycentric-weight
  extension (verified directly against it in the tests); `method="laplacian"` uses the
  classical Nystrom extension (Bengio et al. 2003) of each kept eigenvector individually,
  floored against the (otherwise unbounded) blow-up a near-1 eigenvalue causes in its own
  `1/mu` rescaling, before the same combination the joint solve used at training time.
  Two correctness properties anchor the construction: with independent
  views, both operators' shared (near-)null direction -- the constant vector, which
  every graph-based operator here is blind to -- is explicitly projected out before
  solving (matching `SpectralEmbedding`'s own `drop_first=True`), rather than left in to
  produce a spurious, numerically unstable top component; and each view is restricted to
  its own `n_operator_components` smallest-eigenvalue directions before the joint solve
  (the same truncation every spectral method already does, and effectively this class's
  regularisation strength -- the untruncated limit hands each view as many free
  directions as training points, which, like *any* unregularised multivariate CCA at
  that dimensionality-to-sample-size ratio, fabricates spurious cross-view correlation
  from pure noise). With that in place, `ManifoldCCA` on two duplicated views reduces
  *exactly* (verified to within numerical precision) to plain single-view spectral
  embedding of that one view -- the concrete check that this is the natural multiview
  generalisation of `SpectralEmbedding`, not merely a construction that happens to reuse
  its graph. On two views that are different nonlinear (different angular frequency)
  spiral embeddings of one shared 1-D coordinate -- where no linear map from one view's
  ambient coordinates to the other's exists, but a k-NN graph on either still respects
  the shared ordering -- `method="laplacian"` recovers substantially higher held-out
  correlation than plain `MCCA` or `KCCA` with an RBF kernel (see
  `tests/nonparametric/test_manifold_cca.py`). Hessian-LLE and LTSA are not implemented
  (both need a local Hessian/tangent-space estimate per point, meaningfully more involved
  to get right than the graph Laplacian or LLE's reconstruction weights); Isomap-flavoured
  (geodesic) regularisation is already achievable via `KCCA` with a precomputed geodesic
  Gram matrix, so isn't duplicated here. Like `KCCA`, `inverse_transform`/`predict` aren't
  supported (both assume a `(n_features_i, k)` weight, not a `(n_train_samples, k)`
  transductive embedding).
- `GraphicalLassoCCA`: `MCCA` with each view's within-view covariance block replaced by
  `sklearn.covariance.GraphicalLasso`'s (or, with `alpha=None`, `GraphicalLassoCV`'s)
  L1-penalised sparse-precision estimate's implied covariance -- every other within-view
  regularisation already here (`MCCA`'s own ridge `c`, `CCAR3`'s Ledoit-Wolf shrinkage,
  `TrimmedCCA`'s concentration steps) regularises the covariance directly; this instead
  penalises the *inverse* covariance's off-diagonal entries (each view's own partial
  correlations / conditional independence structure), which is the more natural
  regulariser once a view's feature count approaches or exceeds its sample count -- the
  same high-dimensional regime where `MCCA`'s own docs point to `pca=True` instead. The
  between-view block is untouched (a plain sample cross-covariance, as in `MCCA`); only
  the within-view block each view contributes to the generalised eigenproblem changes.
  Solves directly in each view's original feature space, since the sparse precision
  structure -- inspectable afterwards via `model.precision_` -- is normally the point,
  not a rank-truncated approximation of it.
- `TrimmedCCA`: robust multiview CCA via concentration steps, in the style of Rousseeuw's
  Least Trimmed Squares / Minimum Covariance Determinant. `RANSACCCA`'s random-subset
  search is a good strategy while contamination stays well below its own odds of ever
  drawing a clean-enough candidate, but that degrades near the ~50% breakdown point,
  where a random `min_samples`-sized draw becomes close to a coin flip on being usably
  clean however many trials are tried. `TrimmedCCA` instead starts from a large random
  subset of `h_frac * n` rows and alternates a *select* step (rank every sample by its
  own contribution to `CCAEY`'s exact loss for the current weights, solved via a
  Lagrangian relaxation of the loss's own additive structure, and keep the best `h`) with
  a *refit* step (re-minimise `CCAEY`'s exact loss on just those rows, warm-started via
  L-BFGS-B) -- the classical C-step argument, applied to CCAEY's real objective rather
  than a proxy score, so each step is provably non-increasing in the actual loss. On a
  sign-flip contamination benchmark near 47% contamination, with `h_frac` set close to
  the true clean fraction, `TrimmedCCA` holds at oracle-level held-out correlation
  (~0.72) while `RANSACCCA` degrades to ~0.36-0.48 even with its best-tuned
  `min_samples` and extra trials (see `tests/test_trimmed_cca.py`). `h_frac` is a prior
  on the contamination rate rather than something learned from the data -- like
  `sklearn.covariance.MinCovDet`'s `support_fraction`. Supports any number of views (2
  or more) -- the underlying algebra (mean pairwise cross-covariance and mean
  auto-covariance are already additive over samples, for any number of views) doesn't
  depend on it -- but only `latent_dimensions=1`: past one latent dimension, `CCAEY`'s
  penalty term becomes a genuine matrix-valued quadratic form (rank up to `k(k+1)/2`
  instead of the rank-1 "square of one linear functional" the single-multiplier
  bisection relies on), which would need a different (and considerably heavier)
  optimiser to solve with the same guarantee. Away from the breakdown regime, or when
  more than one latent dimension is needed, `RANSACCCA` matches or beats it directly.
- `RANSACCCA`: the multiview-CCA analogue of `sklearn.linear_model.RANSACRegressor`,
  robust to a different contamination pattern than `HuberCCA`'s. `HuberCCA` downweights
  samples by their *leverage* (combined magnitude across views); that leaves untouched a
  subset of rows whose cross-view *relationship* is wrong (mismatched, corrupted, or drawn
  from an unrelated pattern) while remaining completely ordinary in magnitude within each
  view on its own -- nothing about such a row's norm flags it as unusual, so leverage-based
  reweighting can't see it, and in practice can even make the fit slightly worse. `RANSACCCA`
  instead repeatedly fits a fast closed-form `MCCA` on a random subset of rows, scores each
  candidate by how much of the *full* dataset agrees with it (each sample's own standardised
  cross-view product, the per-sample contribution to the EY reward term's cross-covariance
  trace -- positive for genuine agreement, at or below zero otherwise, which is why
  `residual_threshold` defaults to exactly 0 rather than anything estimated from the data),
  and refits on the best-supported candidate's consensus set. Demonstrated on synthetic data
  with a fraction of rows sign-flipped between views (ordinary magnitude in both views, so
  invisible to `HuberCCA`) in `tests/test_ransac_cca.py`.
- `ProjectionPursuitCCA`: robust multiview CCA via projection pursuit, structurally
  different from `HuberCCA`/`RANSACCCA`/`TrimmedCCA`'s common covariance-based
  machinery -- it never forms a cross- or auto-covariance matrix at all, instead
  searching directly over candidate unit-norm projection directions for the pair (or,
  for more than two views, the tuple averaged pairwise) maximising a robust bivariate
  correlation measure -- the *projection index* -- between the resulting scores. Follows
  the classical projection-pursuit paradigm (Huber 1985) as carried over to CCA by
  Branco, Croux, Filzmoser & Oliveira (2005) and put on firmer statistical footing by
  Alfons, Croux & Filzmoser (2017), whose R package `ccaPP` (Alfons, Croux &
  Filzmoser, 2016) is the reference implementation these projection indices follow.
  Two projection indices are
  available: `projection_index="spearman"` (default, Spearman rank correlation --
  insensitive to an outlier's exact magnitude, only its rank) and `"mcd"` (a
  minimum-covariance-determinant-based correlation via `sklearn.covariance.MinCovDet`,
  cheaper per evaluation than a full robust-covariance-plugin CCA fit since it's only
  ever applied to a 2-dimensional projected scatter). Each direction is parametrised by
  `p_i - 1` unconstrained hyperspherical angles and found by `scipy.optimize.minimize`
  with Powell's method (derivative-free, since a rank-correlation-based objective is
  non-smooth) from `n_restarts` random starting points; further latent dimensions reuse
  the same Gram-Schmidt `deflate` convention `cca_zoo.sparse`'s ALS-based methods use.
  Unlike the other three robust estimators, has no `inlier_mask_` -- its robustness comes
  from the projection index's own insensitivity to extreme values, not from singling out
  individual bad rows.
- `CatBoostCCA`: a third `TreeCCA` backend alongside `XGBoostCCA`/`LightGBMCCA`, using
  [CatBoost](https://catboost.ai/)'s gradient-boosted trees as the per-view encoders. Since
  CatBoost has no in-place "add one tree to this booster" call, each round every component is
  instead replaced by a freshly constructed `CatBoostRegressor(iterations=1, ...)` continued
  from the previous round's model via CatBoost's own `init_model=`, driven by a custom loss
  object that relays the EY gradient as CatBoost's expected `(der1, der2)` pair (`der1 =
  -gradient`, `der2 = -1.0`, matching the unit-Hessian Newton step `XGBoostCCA`/`LightGBMCCA`
  already take). This makes `CatBoostCCA` markedly slower per round than the other two backends
  (CatBoost rebuilds its training pool and recomputes feature-importance statistics on every
  such call) -- reach for it when CatBoost's ordered boosting and symmetric trees are themselves
  the point, not as a faster default. Requires the optional `catboost` package, now part of the
  `tree` extra alongside `xgboost`/`lightgbm`.
- `BaseModel.inverse_transform`: reconstructs each view from *that same view's own*
  latent score (typically `transform`'s output), via a per-view loading matrix fit by
  least squares at training time -- an approximate round trip with `transform`, mirroring
  `sklearn.decomposition.PCA.inverse_transform` (see #195). Distinct from `predict`, which
  combines the *observed* views into one shared consensus score to reconstruct views you
  don't have; `inverse_transform` never mixes information across views. Available on
  every `BaseModel` subclass with no per-model changes needed.
- `BaseModel.predict`: reconstructs every view from whichever views are observed (pass
  `None` for a view to predict, including one you supplied, as a diagnostic). The shared
  latent score is estimated from the observed views' own projections, then each view is
  reconstructed via a per-view loading matrix fit by least squares at training time —
  deliberately not the simpler `scores @ weights.T`, which is only a correct inverse of
  `transform` for CCA when the data happens to be pre-whitened (see #182). Available on
  every `BaseModel` subclass with no per-model changes needed.
- `cca_zoo.model_selection.permutation_test_significance`: permutation test for both
  canonical-correlation significance (per latent dimension) and feature-loading
  significance (per feature, per dimension), following the resampling-based approach used
  in the neuroimaging CCA/PLS literature (Xia et al. 2018; McIntosh & Lobaugh 2004, see
  #130). Since a permuted refit can recover canonical variates in an arbitrary rotated or
  reflected order relative to the true fit, each permutation's loadings are realigned via
  the new `cca_zoo.model_selection.procrustes_rotation` (the SVD solution to the
  orthogonal Procrustes problem) before being compared feature-by-feature.
- `GAMCCA`: nonlinear multiview CCA using a generalized additive model (one B-spline term
  per input feature) as the per-view encoder, trained on the same Eckart-Young objective
  as `TreeCCA` and the `*_EY` models. Conceptually a P-IRLS fit — the same iteration
  structure GAM software such as `mgcv` uses — applied directly to the EY loss, but
  rather than hand-rolling that solve, every view's spline coefficients (every latent
  component, every view, all at once — no per-component or per-view cycling) are updated
  in a single call to `scipy.optimize.minimize(method="trust-krylov")`, a standard
  off-the-shelf trust-region Newton-CG solver, given the loss's exact gradient and an
  exact Hessian-vector product across the whole stacked parameter vector. Smoothing
  strength (`alpha`) is a fixed hyperparameter. Fitted per-feature shape functions are
  inspectable directly via `model.shape_function(view, feature, x)`. Built entirely on
  scikit-learn's own `SplineTransformer`, with `scipy.optimize` doing the Newton-CG solve,
  so no new dependency is required.
- `cca_zoo._utils._ey.random_orthogonal_embedding`: the random-orthogonal
  initial-embedding helper previously private to `TreeCCA` is now a shared EY-loss
  utility, used by `TreeCCA`, `GAMCCA`, and `GaussianProcessCCA`.
- `GaussianProcessCCA`: nonlinear multiview CCA using a Gaussian process with a joint
  (non-additive) ARD-RBF kernel over each view's raw feature vector
  as the per-view encoder, trained on the same Eckart-Young objective as `TreeCCA` and
  `GAMCCA`. Writes each encoder as a fixed cross-kernel basis against a set of basis
  ("inducing") points — every training row by default, or `n_inducing` of them selected
  via `sklearn.cluster.kmeans_plusplus` for datasets too large for exact inference's
  `O(n^3)` cost — and fits the resulting coefficients jointly across every view in a single
  L-BFGS-B run (`scipy.optimize.minimize`), the same algorithm `GaussianProcessRegressor`
  itself uses internally, given the EY loss's exact analytic gradient plus an RKHS-norm
  ridge penalty.
  Kernel hyperparameters are fixed (pass `kernel=` explicitly, or tune externally). Unlike
  `GAMCCA`'s additive splines, a joint kernel can represent a genuine interaction between
  two features of the same view directly. As a Bayesian model, `transform(...,
  return_std=True)` also returns each latent component's posterior standard deviation.
  Built entirely on scikit-learn's own `GaussianProcessRegressor`, `RBF`, `ConstantKernel`
  and `KernelCenterer`, so no new dependency is required.
- `ElasticNetCCA`: sparse linear multiview CCA, trained on the same Eckart-Young
  objective, with an elastic-net penalty on the per-view weights. Fit by cyclic
  coordinate descent — the same algorithm `sklearn.linear_model.ElasticNet` uses for
  ordinary elastic net — but each coordinate's restriction to the EY loss is an exact
  quartic (not the quadratic ordinary least squares gives), solved to its exact global
  minimiser via `cca_zoo._utils._ey.coordinate_descent_ey`. Because every embedding
  stays exactly linear in the raw (centred) view throughout fitting, `model.weights`
  returns real sparse canonical weight vectors, unlike `TreeCCA`/`GAMCCA`/
  `GaussianProcessCCA`, where it raises `NotImplementedError`. Also gained a
  `positive` option, constraining every weight to be non-negative, mirroring
  `sklearn.linear_model.Lasso`/`ElasticNet`'s own `positive=True`.
- `MultiTaskElasticNetCCA`: `ElasticNetCCA` with sklearn's
  `MultiTaskLasso`/`MultiTaskElasticNet` row-group penalty
  ($\sum_j \|W_i[j,:]\|_2$) in place of a plain per-scalar penalty, so a feature is
  either active in every latent dimension or in none, instead of surviving in one
  component and dropping out of another for no principled reason. Since a whole
  row's coefficients are coupled through the EY loss's auto-covariance cross terms
  (unlike ordinary least squares, where a multi-task row is separable across
  tasks), there is no closed-form joint minimiser the way there is for
  `ElasticNetCCA`'s single scalar; each row is instead updated by one step of
  proximal gradient (ISTA) with backtracking line search, accepted only once
  verified to decrease the exact penalised objective, in the new
  `cca_zoo._utils._ey.group_coordinate_descent_ey`.
- `OrthogonalMatchingPursuitCCA`: the EY-loss analogue of
  `sklearn.linear_model.OrthogonalMatchingPursuit` — a fixed per-view sparsity
  budget (`n_nonzero_coefs`) reached by greedy forward selection instead of a
  continuous penalty strength. Features are added one at a time by the same
  residual-correlation criterion classical OMP uses (generalised from a scalar
  to a per-latent-dimension vector, ranked by norm), with the active
  coefficients re-solved to their exact joint (unpenalised) optimum after every
  addition via `ElasticNetCCA`'s own exact quartic coordinate solve. Since the
  EY loss's all-zero embedding is itself a degenerate stationary point (unlike
  ordinary least squares), every view is first warm-started with a small dense
  fit before any view's support is grown from scratch — see
  `cca_zoo._utils._ey.omp_coordinate_descent_ey`'s docstring.
- `StochasticCCAEY`: mini-batch momentum SGD on the same Eckart-Young loss as `CCAEY`, for
  datasets too large for a full-batch gradient evaluation. Fit the way
  `sklearn.linear_model.SGDRegressor` fits a linear model: each epoch, the data is shuffled
  once and split into `batch_size` chunks (`sklearn.utils.gen_batches`), taking one momentum
  gradient step per chunk. A self-contained numpy implementation, with no new dependency
  required.
- `cca_zoo.model_selection.RandomizedSearchCV`: a multiview adapter around
  `sklearn.model_selection.RandomizedSearchCV`, alongside the existing `GridSearchCV`, for
  sampling continuous hyperparameters (e.g. `c` via `scipy.stats.loguniform`) instead of only
  searching a fixed grid.
- `cca_zoo.model_selection.MultiviewWrapper`: the adapter `GridSearchCV`/`RandomizedSearchCV`
  use internally to make a multiview estimator's `fit(views)` look like sklearn's
  `fit(X)` (views horizontally stacked into one array, split back before delegating) is now
  public, so it composes directly with any sklearn model-selection tool -
  `HalvingGridSearchCV`, `cross_val_score`, `cross_validate`, `learning_curve`, `Pipeline`
  - not just the two search classes cca_zoo ships.
- Independent per-view parameter grids: `MultiviewWrapper.set_params` now understands a
  `name__<view index>` suffix (e.g. `c__0`, `c__1`), so `param_grid={"c__0": [...], "c__1":
  [...]}` searches the two views' values independently - sklearn's `ParameterGrid` takes
  their Cartesian product automatically. Previously the only way to sweep a per-view
  parameter was `param_grid={"c": [[0.01, 0.1], [0.5, 0.9]]}`, a fixed list of whole
  per-view vectors that conflates "one candidate" with "one vector per view" and requires
  the user to hand-enumerate any Cartesian product themselves; this is now the documented
  way to tune a per-view hyperparameter in a search. An index not mentioned in the grid
  keeps the estimator's current value for that view rather than requiring every view to be
  listed.
- `cca_zoo.linear.HuberCCA`: a bounded-influence variant of `CCAEY` for data with
  high-leverage outliers. `CCAEY`'s cross- and auto-covariance statistics weight every
  sample equally, so a handful of high-leverage points (whose contribution to a quadratic
  statistic grows with the *square* of their magnitude) can dominate the fit; `HuberCCA`
  reweights each mini-batch sample by a Huber-style factor of its own leverage before
  forming those statistics -- the same bounded-influence mechanism
  `sklearn.linear_model.HuberRegressor` uses against outliers, applied to the EY loss's own
  statistics rather than to a regression residual. The leverage cutoff (`delta`) is a
  multiple of the current batch's own median leverage rather than an absolute threshold, so
  it self-calibrates across `batch_size`/`latent_dimensions` instead of needing per-config
  retuning. Confirmed to substantially outperform `CCAEY` on held-out canonical correlation
  when training data is contaminated by a small fraction of high-leverage points (see
  `tests/test_huber_cca.py`'s outperformance test).

### Fixed

- `SCCAADMM` solved the wrong problem entirely: a reduced-rank-regression-style loss
  `||Xw - target||^2` with the unit-ball constraint on the weight vector `w` itself.
  Reading the actual paper (Suo, Mineiro & Anandkumar 2017, Section 2.2) shows the real
  objective is *linear* in `w` (a covariance to maximise, `w^T X^T target - tau*||w||_1`),
  constrained on the *score* `||Xw||_2 <= 1`, not on `w` -- these coincide only when `X`
  is orthonormal. An intermediate step in this same investigation "fixed" a step-size
  scaling bug in the old (wrong) regression-style objective's primal update, which had
  been causing `nan` divergence at ordinary sample sizes -- a real bug, correctly fixed
  in isolation, but fixing consistency *within* the wrong objective, not the right one.
  Re-implemented following the paper's own linearised-ADMM derivation: since the
  constraint couples `w` to `Xw` through a linear map (not the identity), an ordinary
  ADMM split would need to invert `X^TX` every step, so the augmented Lagrangian's
  quadratic penalty is linearised instead, turning the `w`-update into a single
  proximal-gradient step that is closed-form here (the linear-plus-L1 objective's
  proximal operator is a shifted soft-threshold). Verified directly against first-order
  KKT optimality conditions of the exact constrained problem (no off-the-shelf solver
  handles this reliably -- scipy's `trust-constr` gets stuck at the L1 kink at the
  origin regardless of starting point): the dual variable recovered from any two active
  coordinates agrees to within numerical tolerance, and every zeroed coordinate's
  subgradient residual falls inside `[-tau, tau]`. Adds an `admm_iter` parameter for the
  new inner (per-view, per-outer-iteration) linearised-ADMM loop's iteration cap,
  separate from `max_iter` (the outer across-view loop, matching this module's shared
  convention). `tau`'s default reverts to `0.1` (briefly lowered to `0.01` for the
  interim, wrong-objective fix, no longer needed now the objective itself is correct).
- `CCAR3(highdim=True)` (the default) systematically over-penalised relative to what
  `lambda_` documents: its hand-rolled ADMM solver for the row-group-lasso reduced-rank
  regression subproblem had a factor of 2 missing from its B-update's linear system (the
  gradient of the documented loss `(1/n)||Y-XB||^2` is `(2/n) X^T(XB-Y)`, but the code's
  coefficient matrix and right-hand side both used the `1/n`-scaled versions), silently
  doubling the effective penalty strength. It converged cleanly and passed every existing
  (qualitative) sparsity test, so the bug only surfaced by comparing its objective value
  against an independently-implemented solver: it landed on a different, worse-objective
  stationary point on every problem tested, not merely an under-converged one (raising
  `max_iter`/tightening `tol` made no difference). Fixed by replacing the ADMM solver with
  `sklearn.linear_model.MultiTaskLasso` -- the row-group-lasso subproblem CCAR3 poses is
  exactly `MultiTaskLasso`'s own objective, so this is correct by construction (its
  coordinate descent provably reaches the global optimum of this convex problem) rather
  than a solver that needs independently verifying, and is substantially faster in
  practice. The `rho` (ADMM step-size) parameter is removed, having no equivalent in
  coordinate descent; `max_iter`/`tol` now pass straight through to `MultiTaskLasso`.
- `CCAEY`, `PLSEY`, `HuberCCA`, and their shared base `BaseFullBatchEYModel` defaulted
  `tol=1e-6`, passed straight through to L-BFGS-B as `ftol`. `ftol` is a *relative*
  per-step improvement test, and this loss can pass through slow, shallow stretches of
  genuine descent (e.g. while moving away from a spurious stationary point towards the
  real one) that a loose `ftol` mistakes for convergence -- L-BFGS-B reports success
  either way, so the fit silently stops early rather than raising an error, returning a
  badly wrong result with a suspiciously *low* held-out correlation rather than a visible
  failure. The default is now `tol=1e-8`, which removed the failure in repeated testing on
  a stress-test construction (0/20 failures at `1e-8` vs. a measurable failure rate at
  `1e-6`) without needing any change to initialisation or multi-start. `GPCCA` uses the
  same `ftol` mechanism but showed no evidence of the same failure mode in testing and is
  left unchanged; `GAMCCA` uses `trust-krylov`'s `gtol` (a gradient-norm criterion, not
  `ftol`) and `StochasticCCAEY` checks its own absolute per-epoch objective change, both
  structurally different and also left unchanged pending separate verification.
- `GridSearchCV.cv_results_`'s `param_*` keys carried an internal `estimator__` prefix
  (`param_estimator__c` rather than `param_c`), inconsistent with the unprefixed keys in
  `best_params_` and with the docs' own `cv_results_` examples, which would `KeyError`.
  `cv_results_` (both its `param_*` columns and its `params` list of dicts) is now stripped of
  the prefix the same way `best_params_` already was.
- `GridSearchCV` previously copied over only four hand-picked attributes from the underlying
  `sklearn.model_selection.GridSearchCV` search (`cv_results_`, `best_score_`, `best_params_`,
  `best_estimator_`), silently dropping the rest of sklearn's attribute surface
  (`best_index_`, `scorer_`, `n_splits_`, `refit_time_`, `multimetric_`, ...). Every fitted
  (trailing-underscore) attribute is now forwarded generically, so newer sklearn attributes are
  picked up automatically instead of needing this module updated by hand.

### Changed

- `SCCAPMD`, `SCCAADMM`, `SCCAIPLS`, `SCCASpan`, `WaijenborgCCA`, `ParkhomenkoCCA`, and `SAR`
  move from `cca_zoo.linear` to `cca_zoo.sparse`, alongside the existing EY-loss sparse methods
  (`ElasticNetCCA`, `MultiTaskElasticNetCCA`, `OrthogonalMatchingPursuitCCA`) -- all ten are
  sparse/regularised CCA methods, and belong together regardless of which of the two mechanism
  families (ALS vs. EY-loss coordinate descent) each uses. The old `cca_zoo.linear` import paths
  still work but now emit a `FutureWarning` (via `sklearn.utils.deprecated`) and will be removed
  in a future release; import them from `cca_zoo.sparse` instead.
- `SCCAPMD`, `SCCAADMM`, `SCCAIPLS`, and `SCCASpan` (now living in `cca_zoo.sparse`, see above)
  are further renamed to `PMDCCA`, `ADMMCCA`, `IPLSCCA`, and `SpanCCA`: the `SCCA` ("Sparse CCA")
  prefix is redundant now that these classes are namespaced under `cca_zoo.sparse` itself, and
  bare `PMD`/`ADMM`/`IPLS`/`Span` aren't self-describing as CCA methods on their own, so each
  keeps a `CCA` suffix instead, matching the `<Algorithm/Author>CCA` pattern the rest of the
  module already follows (`ElasticNetCCA`, `WaijenborgCCA`, `ParkhomenkoCCA`). `SpanCCA` now
  shares its literal name with the algorithm it's inspired by (Asteris et al. 2016's own
  "SpanCCA") but remains the same ALS heuristic it always was, not a reimplementation of that
  paper's own low-rank sampling algorithm -- see the class docstring. The old `SCCAPMD`/
  `SCCAADMM`/`SCCAIPLS`/`SCCASpan` names stay importable from `cca_zoo.sparse` (and from
  `cca_zoo.linear`, via the cross-module aliases above) as deprecated aliases.
- `StochasticCCAEY` moves from `cca_zoo.linear` (via `cca_zoo.linear.gradient`) to a new
  `cca_zoo.stochastic` module. Mini-batch fitting is a genuinely different operational regime
  (streaming or out-of-core data) from every other class in `cca_zoo.linear`, which all assume
  the full dataset fits in memory for a single `fit` call, so it gets its own top-level module
  rather than staying folded in among the full-batch EY-loss classes. The old
  `cca_zoo.linear.StochasticCCAEY` import path still works but now emits a `FutureWarning` and
  will be removed in a future release; import it from `cca_zoo.stochastic` instead.
- `cca_zoo.linear.ElasticCCA` is renamed `WaijenborgCCA`, after the paper's own author
  (Waaijenborg 2008), to disambiguate it from `cca_zoo.sparse.ElasticNetCCA` -- a
  different algorithm entirely (an elastic-net penalty on the actual Eckart-Young CCA
  loss, not this class's alternating-regression heuristic), not just a different
  implementation of the same one. `ElasticCCA` stays importable as a deprecated alias.
- `SCCAPMD`'s per-view soft-threshold bisection (`_bisect_threshold`, finding the
  threshold hitting a target L1/L2 ratio) now uses `scipy.optimize.brentq` instead of a
  hand-rolled bisection that unconditionally ran all 50 iterations regardless of how
  quickly it had already converged. `brentq`'s superlinear convergence plus a real
  tolerance-based stop reaches the same root (matches the old fixed-count bisection to
  within 1e-9 across 500 random trials) in far fewer evaluations: 3.65x faster in
  isolation, 1.8x faster for a full `SCCAPMD.fit` in a direct benchmark. No behaviour
  change other than speed.
- `TreeCCA(backend="xgboost"/"lightgbm")` is replaced by two concrete classes,
  `XGBoostCCA` and `LightGBMCCA`, each fixing one gradient-boosting backend. `TreeCCA`
  itself becomes an abstract base class holding the shared Eckart-Young fitting/transform
  recipe and can no longer be instantiated directly; existing code must switch to
  `XGBoostCCA(...)` or `LightGBMCCA(...)`, dropping the `backend=` argument. This is a
  breaking change with no deprecation shim, since `TreeCCA` shipped in `3.1.0` with a
  string `backend=` switch rather than a class per backend, unlike every other
  multi-variant model in the package (e.g. `SCCA` + `PMD`/`ADMM`/`IPLS`/`Span`).
- `CCAEY` now natively supports 2 or more views (previously only `MCCAEY` did, as a thin
  subclass); `MCCAEY` and `MCCA_EY` are now deprecated aliases for `CCAEY` and emit a
  `FutureWarning` on instantiation, since `CCAEY` provides their functionality directly.
- `CCAEY` and `PLSEY` are now fit by full-batch L-BFGS-B (`scipy.optimize.minimize`) instead
  of a hand-rolled mini-batch momentum gradient-descent loop, since the default (and only
  previously well-tested) use of these classes was already full-batch and deterministic; the
  `learning_rate`, `momentum`, and `batch_size` parameters are removed from both. For the
  large-scale, mini-batch use case those parameters used to serve, use the new
  `StochasticCCAEY`.
- `GridSearchCV` and `RandomizedSearchCV` are now themselves `sklearn.base.BaseEstimator`
  subclasses (previously plain classes), so `get_params`/`set_params`/`clone`/`repr` work on the
  search objects too - e.g. `sklearn.base.clone(gs)` before fitting, or nesting one inside
  another meta-estimator.

- Renamed every underscored algorithm-suffix class to drop the underscore, matching sklearn's own
  class-naming convention (`RidgeCV`, `SGDRegressor`, never `Ridge_CV`), with no exceptions:
  `CCA_EY` -> `CCAEY`, `MCCA_EY` -> `MCCAEY`, `PLS_EY` -> `PLSEY`, `DCCA_EY` -> `DCCAEY`,
  `PLS_ALS` -> `PLSALS`, `SCCA_PMD` -> `SCCAPMD`, `SCCA_ADMM` -> `SCCAADMM`,
  `SCCA_IPLS` -> `SCCAIPLS`, `SCCA_Span` -> `SCCASpan`, `DCCA_NOI` -> `DCCANOI`,
  `DCCA_SDL` -> `DCCASDL`. The old underscored names still work but now emit a `FutureWarning`
  (via `sklearn.utils.deprecated`) and will be removed in a future release. This only affects the
  *algorithm*-suffix slot (how a model is fit, e.g. `SCCA` + `PMD`/`ADMM`/`IPLS`/`Span`); the
  *architecture*-prefix slot (what a model is, e.g. `K`/`G`/`T`/`D`/`M`/`Tree` + `CCA`) is otherwise
  untouched, since those already match the abbreviation each method's own literature uses (`DCCA`
  for Deep CCA, `KCCA` for Kernel CCA, etc.) rather than being a naming-convention artifact that
  could be tidied.
- Renamed `GPCCA` -> `GaussianProcessCCA`, matching sklearn's own `GaussianProcessRegressor`/
  `GaussianProcessClassifier` naming rather than abbreviating "Gaussian Process" to `GP` — unlike
  `DCCA`/`KCCA`/`TCCA`/`GCCA`, `GPCCA` isn't an abbreviation inherited from an external paper (it's
  new to this library), so there's no established literature form to preserve, and sklearn's own
  precedent for this exact algorithm spells it out in full. `GAMCCA` and `TreeCCA` keep their
  current names: `GAM` is a self-sufficient term of art in statistics the way `MLP`/`PLS`/`ARD` are
  in sklearn (nobody spells out "Generalized Additive Model" as a model name), and `TreeCCA` is
  already the name given in its own source paper. `GPCCA` still works but now emits a
  `FutureWarning` and will be removed in a future release.
- `CCA_EY` (and `MCCA_EY`, which inherits it) now initialises its weights
  differently from `PLS_EY`, matching each loss's own structure: `PLS_EY`
  keeps a plain, data-independent unit-norm-orthogonal-weight
  initialisation (`cca_zoo._utils._ey.random_orthonormal_weights`), while
  `CCA_EY` now uses a cheap, data-informed initialisation
  (`cheap_orthonormal_projection_weights`) that gives exactly
  unit-variance, uncorrelated *projections* on one mini-batch instead — a
  cheap stand-in for classical CCA's full whitening step, without the
  full-batch cost. As a side effect, `PLS_EY` and `CCA_EY(c=1)` no longer
  fit identical weights from the same seed (only their loss/gradient
  formula is still identical; `tests/linear/test_gradient.py` was updated
  to check that invariant directly instead). Note this initialisation
  change does not, by itself, mitigate the `c=0` divergence risk described
  above — empirically confirmed it does not measurably postpone it either,
  since every later step draws an independent fresh mini-batch — `c` and
  `batch_size` remain the actual remedy for that.

- The package version is now derived from git tags (`hatch-vcs`) instead of a hand-maintained
  `version = "..."` string in `pyproject.toml`. This closes the exact failure mode that
  motivated it: a tag and the shipped version silently disagreeing because someone forgot the
  bump-version PR (or the version happened not to be on PyPI yet, in which case the mismatch
  would previously have published *permanently* under the wrong number). Version-bump PRs going
  forward only need to move `CHANGELOG.md`'s `Unreleased` section into a dated one — there's no
  version field left to edit. Verified: a checkout at a tagged commit builds exactly that
  version; any other commit builds a `<last-tag>.dev<N>+g<hash>` version; a shallow/no-tags
  checkout falls back to a `0.1.dev...` version rather than failing the build outright.
- The PyPI publish job now triggers on a GitHub *Release* being published, not a raw
  `git push --tags`. Drafting a Release (`gh release create` or the UI) is a more deliberate,
  visible act than pushing a tag, and is also the trigger most current guidance recommends for
  Trusted Publishing. The job also now runs behind a `pypi` GitHub Environment for an
  independent required-reviewer approval gate — **the environment itself must still be created
  with a required reviewer under Settings > Environments**, since protection rules aren't
  configurable from a workflow file.
- Added a publish-job step that re-derives the version from the built sdist's filename and
  compares it to the release tag, failing loudly (before any upload) if they disagree, as a
  second, independent check behind the `hatch-vcs`/tag-based versioning above.
- `CITATION.cff`'s `version`/`date-released` fields were stale at `3.0.0` (never updated across
  the `3.1.0` or `3.2.0` releases) — bumped to match. These fields can't be derived automatically
  (Zenodo metadata, not a build artifact), so they stay a manual step in the release checklist.

### Removed

- `PLSALS`, the ALS/power-iteration variant of `PLS`, is dropped outright with no deprecated
  alias -- unlike every other class touched by this release's `cca_zoo.sparse`/
  `cca_zoo.stochastic` moves. It had no sparsity, no ridge regularisation, and no behaviour
  its closed-form `PLS` counterpart doesn't already cover exactly, so there was nothing left
  for it to alias.

## [3.2.0] - 2026-08-04

### Added

- `GFA` (Group Factor Analysis): a third probabilistic CCA backend, ported faithfully from the
  reference R package [`CCAGFA`](https://github.com/cran/CCAGFA) (Klami, Virtanen & Kaski,
  2013) — the update equations are transliterated directly from that source. Unlike
  `VariationalBayesCCA`'s single ARD parameter shared across every view per latent dimension,
  `GFA` gives each view its **own** ARD precision per dimension, so "shared" vs. "private"
  latent structure is emergent from the fitted per-view relevance
  (`view_relevance_`) rather than a fixed split. Inference is closed-form coordinate-ascent
  variational Bayes with no dependency beyond numpy/scikit-learn — no numpyro/jax needed, unlike
  the other two classes, so it's always available regardless of the `[probabilistic]` extra.
  Dynamic dimensionality pruning (`drop_k`, `n_components_`) matches the R package's `dropK`
  default. Deliberately omits the R package's optional rotation-optimization step (an
  optimization-path speedup, not a model change) rather than risk porting it without a
  reference R run to verify against.
- `VariationalBayesCCA`: probabilistic CCA fit via mean-field stochastic variational inference
  (numpyro SVI), a much cheaper alternative to `ProbabilisticCCA`'s full NUTS MCMC. Adds a
  hierarchical automatic relevance determination (ARD) prior shared across views, giving
  automatic latent-dimensionality selection via the new `ard_relevance_` attribute instead of a
  `GridSearchCV` sweep over `latent_dimensions`. This is the "VB-CCA" (Wang 2007) previously
  only cited, not implemented, by `ProbabilisticCCA`'s docstring.
- `log_likelihood()` on `GFA`, `ProbabilisticCCA`, and `VariationalBayesCCA`: the marginal
  log-likelihood of held-out data with the shared latent variable integrated out, evaluated
  jointly across the concatenation of all views (not per view) so it correctly captures the
  cross-view covariance induced by the shared latent structure. Computed via the Woodbury
  identity and matrix determinant lemma (verified against a brute-force
  `scipy.stats.multivariate_normal` computation to machine precision). This is the
  statistically proper Bayesian model-fit criterion, complementing (not replacing) `score()`,
  which every model in the package shares for `GridSearchCV` consistency.

### Fixed

- `ProbabilisticCCA.score()` returned `nan` and `.get_factor_loadings()` silently returned only
  the first view's loadings: both are inherited from `BaseModel`, which assumes `transform()`
  returns one array per view, but these joint-latent models return a single shared-z array.
  Fixed for both `ProbabilisticCCA` and the new `VariationalBayesCCA` via a shared mixin that
  correlates each view's own posterior-mean projection instead.
- `ProbabilisticCCA`'s docs referenced a `model.mcmc_` attribute for ArviZ diagnostics that
  `fit()` never actually set; now stored.
- `ProbabilisticCCA.weights_` was silently biased toward zero by the model's rotational
  symmetry ($z \to zR$, $W_i \to W_i R$ for shared orthogonal $R$ leaves the likelihood
  unchanged): different NUTS draws settle on different rotations, and averaging them
  un-aligned partially cancels rather than reinforces the signal. Measured on a synthetic
  check: a rotation-invariant coherence ratio of `||mean(W)||²` vs `mean(||W||²)` across draws
  (1.0 if every draw agrees on a rotation) was 0.81 before the fix. `fit()` now aligns every
  draw's loadings (and that draw's own `z`) to a common reference via generalized Procrustes
  analysis (`align_posterior_rotation`) before computing `weights_`; the same check now gives
  0.99. `VariationalBayesCCA` doesn't need this — its mean-field SVI posterior already
  collapses onto a single rotation (checked: ratio 0.9996 without any correction).
- `CCA_EY` (and `MCCA_EY`, which shares its `fit()`) whitened the *entire* dataset with a
  full-batch SVD (`svd_whiten`) before doing any mini-batch gradient descent — an O(full-dataset)
  step fundamentally at odds with these classes being the large-scale/streaming member of the
  Eckart-Young family. `PLS_EY`, `TreeCCA`, and `DCCA_EY` already applied the shared EY loss
  directly to raw mini-batches with no such step; `CCA_EY` now does too. Removing the whitening
  step surfaced a real numerical-stability gap: gradient descent on the raw, unregularised loss
  can diverge to `nan` when a mini-batch's `batch_size` doesn't comfortably exceed
  `n_features` (nothing then bounds the weights in the mini-batch's near-null directions).
  `CCA_EY` keeps its `c` ridge parameter to address this — reworked into a blend, in the
  unconstrained/stochastic setting, of the same canonical-ridge idea `rCCA` already uses
  ($(1-c)X^\top X + cI$): `c=0` is exactly the original, unregularised objective (default,
  unchanged for well-conditioned data), and `c=1` is exactly `PLS_EY`'s objective, so `c`
  continuously blends `CCA_EY` towards `PLS_EY`'s (empirically more stable) loss. `PLS_EY` is
  now implemented as a thin `CCA_EY` subclass with `c` fixed at `1` (not exposed in its own
  `__init__`), mirroring how `CCA`/`PLS` are thin `rCCA` subclasses with `c` fixed at `0`/`1`.
  The blended gradient is verified against finite differences and, at its `c=0`/`c=1` endpoints,
  against bit-for-bit exact matches with the pre-existing unregularised EY gradient and with
  `PLS_EY`'s own independently verified gradient.
- `CCAR3`'s ADMM row-sparse solver returned `B`, the smooth ADMM working variable, instead of
  `Z`, the variable its group soft-threshold is actually applied to. `B` only converges
  *towards* `Z` within `tol` in an aggregate Frobenius sense, so individual rows of `B` stay
  generically nonzero (just small) well past the default `tol=1e-4` — meaning `lambda_` had no
  visible effect on row sparsity across several orders of magnitude at the library's own
  default tolerance, even though `Z` was correctly sparse throughout. Verified against the
  reference R implementation (`jameschapman19/ccar3`), which hits the same issue and works
  around it with a post-hoc absolute threshold; returning `Z` directly is more robust since it's
  exactly sparse by construction regardless of how tight `tol` happens to be. Added a regression
  test at the library's default `tol` (the pre-existing sparsity test used `tol=1e-8`, tight
  enough to mask the bug).

## [3.1.0] - 2026-08-03

### Added

- `CCAR3`: canonical correlation analysis via reduced-rank regression, ported from the
  reference R package `ccar3` (Donnat & Tuzhilina, 2024). Supports both a closed-form
  low-dimensional solver and an ADMM-solved row-sparse high-dimensional solver.
- `TreeCCA`: nonlinear multiview CCA using gradient-boosted trees (XGBoost or LightGBM) as
  the per-view encoders, trained via the same Eckart-Young objective as the `*_EY` models.
- `GRCCA`, `PartialCCA`, `DMCCA`, `DGCCA` restored — these were present in `v2.6.0` but lost
  during the `v3.0.0` rewrite.
- `tests/test_sklearn_compat.py`: every `BaseModel` subclass in the package is now checked
  against four of scikit-learn's own `estimator_checks` (constructor purity, `get_params`/
  `set_params` round-tripping, `repr`).
- A `_parameter_constraints` + `_validate_params()` mechanism (sklearn's own pattern):
  `BaseModel` validates `latent_dimensions`/`center` for every model, and several models
  (`rCCA`, `MCCA`, `GCCA`, `TCCA`, `CCAR3`) validate their own documented-range parameters
  (e.g. a ridge parameter in `[0, 1]`). Invalid parameters now raise a clear error at `fit()`
  instead of failing deep inside the linear algebra.
- CI now also runs the `deep`/`probabilistic`/`tree` test suites (previously only ever run
  locally, never in CI) and doctests for those modules.

### Fixed

- Mathematical rendering was broken across the entire documentation site — docstrings used
  Sphinx/RST math syntax (`.. math::`, `:math:`) on a MkDocs/Markdown site that only
  understands `$...$`. Every model's docstring math now actually renders.
- `sklearn.utils.Tags`/`__sklearn_tags__` (required as of scikit-learn 1.6) replaced the
  removed `_more_tags()`, fixing CI on recent scikit-learn versions.
- A stale `mypy` target Python version caused spurious CI failures against recent numpy
  stub files.
- The documented deep-module training example didn't actually work (wrong `Dataset` batch
  shape).
- `TreeCCA`'s docstring cited the general Eckart-Young paper instead of its own paper;
  now cites Chapman (2026), arXiv:2607.27027.
- `CCA_EY`/`PLS_EY`/`MCCA_EY` correctness fixes; the underlying Eckart-Young machinery is
  now shared with `TreeCCA` and the deep `*_EY` models rather than duplicated.

### Changed

- Dependency management modernized to `uv`: PEP 735 `[dependency-groups]` (replacing the
  deprecated `[tool.uv] dev-dependencies`), a committed `uv.lock`, and a CI/docs workflow
  built around `uv sync --locked`.
- `README.md` rewritten to reflect the current method list, install instructions, and
  badges.
- Every public class's docstring now includes a literature `References:` section.
- `svd_whiten` (used by `rCCA` and `CCA_EY`) now takes a covariance-eigendecomposition
  path when `n_samples >= n_features`, instead of always computing the full thin SVD of
  `X`. Avoids allocating an `n x p` matrix for tall data — up to ~14x faster on a
  54,000 x 392 benchmark. No change to public API or results.
- Project logo and favicon replaced with a hand-authored two-ring mark (previously a large
  auto-traced SVG); the favicon is now a proper multi-resolution `.ico` generated from it.

### Removed

- `examples/` deleted: it wasn't wired into the docs site or CI, and every script had been
  silently broken for a while (`matplotlib` was never a declared dependency). Its content
  overlapped with the maintained `docs/user-guide/*.md` pages.
- `benchmark/` deleted: the same problem as `examples/` (undeclared `matplotlib`/`seaborn`/
  `pandas` dependencies, plus API drift against the current `CCA_EY` signature), and even
  once made runnable its output wasn't linked from anywhere a user would see it, and it only
  compared runtime — not accuracy — for 3 of the ~24 model classes in the package.
- Redundant hand-written `get_params`/`set_params` roundtrip tests, now that
  `tests/test_sklearn_compat.py` covers every model generically.
- `.readthedocs.yaml` (dead Sphinx config; docs are built with MkDocs).
- Unused favicon-generator output (`favicon-16x16.png`, `favicon-32x32.png`,
  `apple-touch-icon.png`, `android-chrome-*.png`, `site.webmanifest`) — never referenced by
  `mkdocs.yml`, `README.md`, or the built site.

## [3.0.0] - 2026-03-07

Initial 3.0.0 rewrite. See git history prior to this file's addition for details.
