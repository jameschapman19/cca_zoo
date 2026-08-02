"""Ensures every public API symbol has a docs/api/*.md entry.

This directly guards against the failure mode that let several v2.6.0
models go undocumented (and briefly GRCCA/PartialCCA/DMCCA/DGCCA too): a
class gets added to a module's ``__all__`` but nobody remembers to add a
matching mkdocstrings ``::: `` line to the corresponding API reference
page. docs/api/*.md isn't auto-generated from ``__all__`` (it's hand
-curated into meaningful categories with headers), so this test is the
mechanical backstop that catches the gap instead.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

DOCS_API_DIR = Path(__file__).parent.parent / "docs" / "api"

# Importable module name -> docs/api/*.md filename. Usually the same name;
# model_selection is the one exception (hyphenated filename).
MODULE_DOC_PAGES = {
    "cca_zoo.linear": "linear.md",
    "cca_zoo.deep": "deep.md",
    "cca_zoo.nonparametric": "nonparametric.md",
    "cca_zoo.probabilistic": "probabilistic.md",
    "cca_zoo.tree": "tree.md",
    "cca_zoo.datasets": "datasets.md",
    "cca_zoo.model_selection": "model-selection.md",
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
