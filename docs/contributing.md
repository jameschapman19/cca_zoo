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
- **Google-style docstrings** on all public classes and methods with Args, Returns, Raises,
  and Example sections.
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
   `transform`, `fit_transform`, `score`, `pairwise_correlations`, `get_factor_loadings`,
   and correct sklearn `get_params`/`set_params`/tags for free — implement `fit` only.
3. Add Google-style docstrings including the mathematical objective and reference(s).
4. If any constructor parameter has a documented range (e.g. a ridge parameter in
   `[0, 1]`), declare it in `_parameter_constraints` (merging in the parent class's, e.g.
   `{**BaseModel._parameter_constraints, "c": RIDGE_PARAMETER}` — see
   `cca_zoo/_utils/_param_constraints.py` for shared constraint fragments and
   `cca_zoo/linear/_mcca.py` for an example). This is optional but recommended: it turns
   an invalid parameter into a clear error at `fit()` time instead of a cryptic failure
   deep in the linear algebra.
5. Export from the subpackage's `__init__.py` and add to `__all__`. Doing this is also
   what gets your model automatically covered by `tests/test_sklearn_compat.py`'s
   generic sklearn-estimator-contract checks (`get_params`/`set_params` round-tripping,
   `repr`, init purity) — no per-model test needed for that part.
6. Write tests in `tests/<subpackage>/test_mymodel.py` covering, at minimum: `fit`
   completing without error, `transform`/`fit_transform` output shapes, `score` shape and
   value range, and — where a closed-form or known-correct reference solution exists — a
   correctness check against it (see `tests/linear/test_eigendecomposition.py` for the
   established pattern). If you added `_parameter_constraints`, add a rejection test per
   constraint (see `tests/linear/test_parameter_constraints.py`).
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
