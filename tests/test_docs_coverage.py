"""Every public name has an API reference entry."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from cca_zoo._base import BaseModel

DOCS_API_DIR = Path(__file__).parent.parent / "docs" / "api"

MODULE_DOC_PAGES = {
    "cca_zoo.linear": "linear.md",
    "cca_zoo.metrics": "metrics.md",
    "cca_zoo.deep": "deep.md",
    "cca_zoo.nonparametric": "nonparametric.md",
    "cca_zoo.probabilistic": "probabilistic.md",
    "cca_zoo.tree": "tree.md",
    "cca_zoo.datasets": "datasets.md",
    "cca_zoo.model_selection": "model-selection.md",
    "cca_zoo.preprocessing": "preprocessing.md",
    "cca_zoo.sparse": "sparse.md",
    "cca_zoo.stochastic": "stochastic.md",
}


def _documented_symbols(doc_path: Path) -> set[str]:
    """Extract the trailing symbol name from every mkdocstrings ``::: `` line."""
    text = doc_path.read_text()
    refs = re.findall(r"^::: (\S+)$", text, flags=re.MULTILINE)
    return {ref.rsplit(".", 1)[-1] for ref in refs}


@pytest.mark.parametrize("module_name", sorted(MODULE_DOC_PAGES))
def test_all_public_symbols_are_documented(module_name: str) -> None:
    """Every name in a module's __all__ has a `::: ` entry in its API doc page."""
    module = importlib.import_module(module_name)
    public_names = getattr(module, "__all__", [])
    if not public_names:
        pytest.skip(
            f"{module_name}.__all__ is empty (optional dependency not installed)"
        )

    doc_path = DOCS_API_DIR / MODULE_DOC_PAGES[module_name]
    documented = _documented_symbols(doc_path)

    missing = [name for name in public_names if name not in documented]
    assert not missing, (
        f"{module_name}.__all__ contains names with no docs/api/"
        f"{MODULE_DOC_PAGES[module_name]} entry: {missing}. "
        "Add a `::: ...` line for each, grouped under the relevant heading."
    )


def _members_list_for(doc_path: Path, dotted_path: str) -> list[str]:
    """Extract the mkdocstrings ``members:`` list following a ``::: <path>`` line."""
    lines = doc_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"::: {dotted_path}":
            break
    else:
        raise AssertionError(f"No '::: {dotted_path}' directive found in {doc_path}")

    members = []
    in_members = False
    for line in lines[i + 1 :]:
        stripped = line.strip()
        if stripped == "members:":
            in_members = True
            continue
        if in_members:
            if stripped.startswith("- "):
                members.append(stripped[2:].strip())
                continue
            break
        if stripped.startswith("::: ") or stripped == "---":
            break
    return members


def test_basemodel_members_list_matches_public_api() -> None:
    """The API docs' `members:` list for BaseModel names every public method."""
    doc_path = DOCS_API_DIR / MODULE_DOC_PAGES["cca_zoo.linear"]
    documented = set(_members_list_for(doc_path, "cca_zoo._base.BaseModel"))
    # sklearn adds set_<method>_request metadata-routing setters itself.
    actual = {
        name
        for name in vars(BaseModel)
        if not name.startswith("_") and not re.fullmatch(r"set_\w+_request", name)
    }

    missing = actual - documented
    assert not missing, (
        f"BaseModel has public member(s) {missing} not listed in "
        f"{doc_path}'s `members:` block. Add them there."
    )
    extra = documented - actual
    assert not extra, (
        f"{doc_path}'s `members:` block lists {extra}, which are no "
        "longer public members defined directly on BaseModel. Remove them."
    )
