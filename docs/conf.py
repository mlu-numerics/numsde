"""Sphinx configuration for the NumSDE documentation."""

import os
from importlib.metadata import PackageNotFoundError, version

project = "NumSDE"
author = "Raphael Kruse"
copyright = "2026, Raphael Kruse @ MLU Halle-Wittenberg"

try:
    release = version("numsde")
except PackageNotFoundError:  # pragma: no cover
    release = "unknown"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",  # Markdown (MyST) pages and executable Jupyter notebooks
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",  # first entry: autosummary writes .rst stubs
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints"]
nitpicky = True
nitpick_ignore = [
    ("py:class", "array"),
    ("py:class", "optional"),
    ("py:class", "Any"),
    ("py:class", "numsde._backend.rng.SeedLike"),
]

# -- MyST and notebooks ------------------------------------------------------
myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300

# -- API documentation -------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"of", "or", "optional", "default", "array"}

# -- Bibliography ------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
if os.environ.get("NUMSDE_DOCS_OFFLINE"):  # build without network access
    intersphinx_mapping = {}
    nitpick_ignore_regex = [
        (r"py:.*", r"(python:)?(numpy|scipy|types|abc|list|int|str|tuple)(\..*)?")
    ]

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"NumSDE {release}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/mlu-numerics/numsde",
    "navbar_align": "left",
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "mlu-numerics",
    "github_repo": "numsde",
    "github_version": "main",
    "doc_path": "docs",
}
