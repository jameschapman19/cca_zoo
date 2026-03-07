# Contributing

Contributions are welcome. This guide covers the development setup, coding standards, and the
pull request process.

---

## Development setup

CCA-Zoo uses [uv](https://docs.astral.sh/uv/) in CI, but you can use pip for local development.

### 1. Clone and install

```bash
git clone https://github.com/jameschapman19/cca_zoo.git
cd cca_zoo
pip install -e ".[dev]"
```

For documentation development:

```bash
pip install -e ".[dev,docs]"
```

### 2. Run tests

```bash
pytest -m "not slow"             # fast tests only (no torch / numpyro required)
pytest -m slow                   # deep and probabilistic tests (requires extras)
pytest --cov=cca_zoo             # with coverage report
```

### 3. Lint and format

```bash
ruff check .                     # lint
ruff format --check .            # format check
ruff format .                    # auto-format
```

### 4. Type checking

```bash
mypy cca_zoo
```

### 5. Build docs locally

```bash
mkdocs serve                     # live-reload preview at http://127.0.0.1:8000
mkdocs build                     # build static site into site/
```

---

## Coding standards

All contributions must comply with the standards defined in `CLAUDE.md`. Key points:

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
2. Inherit from `BaseModel` (linear/nonparametric) or `BaseDeep` (deep).
3. Add Google-style docstrings including the mathematical objective and reference(s).
4. Export from the subpackage's `__init__.py` and add to `__all__`.
5. Write tests in `tests/<subpackage>/test_mymodel.py` covering all required test cases
   (see `CLAUDE.md` for the full list).
6. Open a pull request against `main`.

---

## Pull request guidelines

- Keep PRs focused — one feature or fix per PR.
- Include tests for all new/changed behaviour.
- Ensure `ruff check`, `ruff format --check`, `mypy cca_zoo`, and `pytest -m "not slow"` all
  pass before requesting review.
- Reference any related issues in the PR description.

---

## Reporting issues

Use [GitHub Issues](https://github.com/jameschapman19/cca_zoo/issues) to report bugs or
request features. Please include:

- A minimal reproducible example
- The version of cca-zoo (`python -c "import cca_zoo; print(cca_zoo.__version__)"`)
- Your Python version and OS
