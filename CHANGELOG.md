# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

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
