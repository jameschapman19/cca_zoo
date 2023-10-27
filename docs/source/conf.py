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
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

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
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/cca-zoo-logo.svg"
html_favicon = "_static/cca-zoo-logo.svg"
html_sourcelink_suffix = ""
html_last_updated_fmt = ""  # to reveal the build date in the pages meta

html_theme_options = {
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jameschapman19/cca_zoo",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cca-zoo/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "logo": {
        "text": "CCA-Zoo",
        "image": "_static/cca-zoo-logo.svg",
    },
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "navigation_with_keys": False,
}


html_context = {
    "github_user": "jameschapman19",
    "github_repo": "cca_zoo",
    "github_version": "main",
    "doc_path": "docs/source",
}
