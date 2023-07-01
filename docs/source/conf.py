# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from types import ModuleType
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CCA-Zoo"
copyright = "2023, James Chapman"
author = "James Chapman"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

sphinx_gallery_conf = {
    "doc_module": "cca-zoo",
    "examples_dirs": "examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "ignore_pattern": "__init__.py",
}

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("http://scikit-learn.org/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
    "pytorch_lightning": (
        "https://pytorch-lightning.readthedocs.io/en/stable/index.html#",
        None,
    ),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://numpyro.readthedocs.io/en/latest/", None),
    "jaxlib": ("https://jax.readthedocs.io/en/latest/", None),
}

autodoc_default_options = {"members": True, "inherited-members": True}

# generate autosummary even if no references
autosummary_generate = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
