## Summary

<!-- What does this PR do, and why? Link any related issue (e.g. "Fixes #123"). -->

## Type of change

- [ ] Bug fix
- [ ] New model / feature
- [ ] Documentation
- [ ] Refactor / internal change
- [ ] Other (describe above)

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy cca_zoo` passes
- [ ] `uv run pytest -m "not slow"` passes locally
- [ ] Added or updated tests for all new/changed behaviour
- [ ] Added/updated docstrings and, for new models, an entry in `docs/api/*.md`
- [ ] Updated `CHANGELOG.md` if this is user-facing

<!--
Adding a new model? See docs/contributing.md#adding-a-new-model for the full
checklist (BaseModel/BaseDeep inheritance, _parameter_constraints, exports,
docs, tests).
-->
