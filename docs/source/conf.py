# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable without installing it
sys.path.insert(0, os.path.abspath("../../src"))

project = "TensorRSVD"
copyright = "2026, Paul Virally"
author = "Paul Virally"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 3,
    "titles_only": False,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autoclass_content = "both"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pylops": ("https://pylops.readthedocs.io/en/stable/", None),
}
