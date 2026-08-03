# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `VariationalBayesCCA`: probabilistic CCA fit via mean-field stochastic variational inference
  (numpyro SVI), a much cheaper alternative to `ProbabilisticCCA`'s full NUTS MCMC. Adds a
  hierarchical automatic relevance determination (ARD) prior shared across views, giving
  automatic latent-dimensionality selection via the new `ard_relevance_` attribute instead of a
  `GridSearchCV` sweep over `latent_dimensions`. This is the "VB-CCA" (Wang 2007) previously
  only cited, not implemented, by `ProbabilisticCCA`'s docstring.
- `log_likelihood()` on both `ProbabilisticCCA` and `VariationalBayesCCA`: the marginal
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
