# Contributing

Contributions are welcome. This guide covers the development setup, coding standards, and the
pull request process.

---

## Development setup

CCA-Zoo uses [uv](https://docs.astral.sh/uv/) for dependency management, both in CI and for
local development. A `uv.lock` is committed so everyone (and CI) resolves the exact same
dependency versions.

### 1. Clone and install

```bash
git clone https://github.com/jameschapman19/cca_zoo.git
cd cca_zoo
uv sync --group dev --locked
```

`uv sync` creates `.venv` for you; prefix commands with `uv run`, or `source .venv/bin/activate`
first. For documentation development:

```bash
uv sync --extra docs --locked
```

For a specific optional extra (e.g. to work on the deep module):

```bash
uv sync --group dev --extra deep --locked
```

If you don't want to use uv, `pip install -e ".[dev]"` also works, but won't use the lockfile.

### 2. Run tests

```bash
uv run pytest -m "not slow"      # fast tests only (no torch / numpyro / xgboost required)
uv run pytest -m slow            # deep, probabilistic, and tree tests (requires extras)
uv run pytest --cov=cca_zoo      # with coverage report
```

### 3. Lint and format

```bash
uv run ruff check .              # lint
uv run ruff format --check .     # format check
uv run ruff format .             # auto-format
```

Optionally, install [pre-commit](https://pre-commit.com/) to run these (plus mypy) automatically
on every commit:

```bash
uvx pre-commit install
```

### 4. Type checking

```bash
uv run mypy cca_zoo
```

### 5. Build docs locally

```bash
uv run mkdocs serve               # live-reload preview at http://127.0.0.1:8000
uv run mkdocs build --strict      # build static site into site/
```

---

## Coding standards

All contributions must comply with the following:

- **Python ≥ 3.10 only.** Use `X | Y` unions, `list[x]`/`dict[x]`/`tuple[x]` generics.
- **Google-style docstrings**, short and factual. A public class docstring has a one-line
  summary, the objective and method in a few sentences (maths where it defines the model),
  then `Args` (each ending "Default is X."; "Per-view." for parameters that take a scalar or
  one value per view), `Attributes` (the fitted attributes specific to the model; the common
  ones are on `BaseModel`), `References` and a runnable `Examples` section. `fit` docstrings take
  the form "Fit the model." with `Args`, `Returns: self.` and only model-specific `Raises`.
  Private helpers get a line or two. Design rationale, history and comparisons belong in the
  user guide, not the docstring.
- **Full type annotations** — `mypy --strict` must pass cleanly.
- **No `try/except`** — write code that does not need them.
- **No `print`** — use `logging` if diagnostic output is needed.
- **100% test coverage** — every new code path needs a test.
- **No `# pragma: no cover`** — this is banned.

---

## Adding a new model

1. Create the implementation file in the appropriate subpackage
   (e.g. `cca_zoo/linear/_mymodel.py`).
2. Inherit from `BaseModel` (linear/nonparametric) or `BaseDeep` (deep). This gets you
   `transform`, `fit_transform`, `predict`, `inverse_transform`, `score`, and correct
   sklearn `get_params`/`set_params`/tags for free. Implement `fit`, starting with
   `views = self._setup_fit(views)` and ending with `return self._finish_fit(views)`, and
   set `weights_` for a linear model; a nonlinear one overrides
   `_transform_view(view, centred)`, its per-view encoder, which every other method goes
   through. `_finish_fit` records what `predict`, `inverse_transform` and
   `feature_importances_per_view_` need, so the model does not keep its training data.
   Override `_feature_importances(views)` if the model has a native importance (one
   non-negative array per view); otherwise permutation importance is used.
   An iterative model reports `n_iter_` and warns with sklearn's `ConvergenceWarning`
   when it stops at `max_iter`.
3. Add Google-style docstrings including the mathematical objective and reference(s).
4. Declare every constructor parameter in `_parameter_constraints`, merging in the
   parent class's (e.g. `{**BaseModel._parameter_constraints, "shrinkage": RIDGE_PARAMETER}`;
   `cca_zoo/_utils/_param_constraints.py` has shared fragments). An invalid value then
   fails clearly at `fit()`, as in sklearn.
5. Export from the subpackage's `__init__.py` and add to `__all__`. This is what puts the
   model under the generic tests, so it needs no tests of its own for anything they check:
   - `tests/test_estimator_checks.py` runs scikit-learn's estimator checks on it through
     a two-view adapter: input validation, fitted-state errors, pickling, cloning,
     idempotent refits, invariance to sample order, and more.
   - `tests/test_model_contract.py` checks the multiview contract: `transform`,
     `predict` and `inverse_transform`, the number and shapes of views, that every
     parameter is validated, `center=False`, `feature_importances_per_view_`,
     convergence warnings, and that the model
     recovers a strong shared signal at its defaults.
   If a check cannot apply to the model, add it to that file's expected failures with
   the reason.
6. Test only what is specific to the model in `tests/<subpackage>/test_mymodel.py`:
   agreement with a known answer where one exists (a closed form, another model it
   reduces to, a brute-force optimum), what its parameters do, and the behaviour that
   motivates it (robustness to outliers, sparsity, an interaction it captures). Do not
   repeat the generic checks.
7. Add a `::: cca_zoo.<subpackage>.MyModel` entry to the relevant `docs/api/*.md` page —
   `tests/test_docs_coverage.py` enforces this.
8. Open a pull request against `main`.

---

## Pull request guidelines

- Keep PRs focused — one feature or fix per PR.
- Include tests for all new/changed behaviour.
- Ensure `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy cca_zoo`, and
  `uv run pytest -m "not slow"` all pass before requesting review.
- Reference any related issues in the PR description.

---

## Reporting issues

Use [GitHub Issues](https://github.com/jameschapman19/cca_zoo/issues) to report bugs or
request features. Please include:

- A minimal reproducible example
- The version of cca-zoo (`python -c "import cca_zoo; print(cca_zoo.__version__)"`)
- Your Python version and OS
