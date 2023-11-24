# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
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
    "sphinx_favicon",
]

# Produce `plot::` directives for examples that contain `import matplotlib` or
# `from matplotlib import`.
numpydoc_use_plots = True
# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "templates",
    "includes",
    "themes",
    "joss",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

htmllogos_path = ["logos"]

sphinx_gallery_conf = {
    "doc_module": "cca-zoo",
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "ignore_pattern": "__init__.py",
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
}

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "skops": ("https://skops.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://numpyro.readthedocs.io/en/latest/", None),
    "jaxlib": ("https://jax.readthedocs.io/en/latest/", None),
    "lightning": ("https://pytorch-lightning.readthedocs.io/en/latest/", None),
}

# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    # "show-inheritance": True,
    "inherited-members": True,

}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = ".rst"

# The main toctree document.
root_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "logos/cca-zoo-logo.svg"
html_favicon = "logos/cca-zoo-logo.svg"
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
        "image": "logos/cca-zoo-logo.svg",
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
    "doc_path": "docs",
}
