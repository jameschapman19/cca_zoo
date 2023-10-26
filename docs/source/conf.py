# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))

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
exclude_patterns = ["_build", "_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

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
    "sklearn": ("https://scikit-learn.org/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://numpyro.readthedocs.io/en/latest/", None),
    "jaxlib": ("https://jax.readthedocs.io/en/latest/", None),
    "lightning": ("https://pytorch-lightning.readthedocs.io/en/latest/", None),
}

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "inherited-members": True,
    "autodoc_mock_imports": ["torch" "sklearn", "lightning", "jax", "jaxlib"],
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "show_nav_level": 1,
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/jameschapman19/proxtorch",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "jameschapman19",
    "github_repo": "proxtorch",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
html_logo = "cca-zoo-logo.png"
html_favicon = "favicon.ico"
