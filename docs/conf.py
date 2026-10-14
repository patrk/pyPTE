"""Sphinx configuration for the pyPTE documentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the tutorials import the simulation models and figure style from examples/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "_ext"))

from pyPTE import __version__  # noqa: E402

project = "pyPTE"
author = "Patryk Mikulski"
copyright = "2016–2026, Patryk Mikulski"
release = __version__
version = __version__

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "tutorials"]
myst_enable_extensions = ["dollarmath", "colon_fence"]

autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_validation_checks: set[str] = set()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "mne": ("https://mne.tools/stable", None),
}


sphinx_gallery_conf = {
    "examples_dirs": ["tutorials"],
    "gallery_dirs": ["tutorial"],
    # every script is a page; there is no plot_ naming convention to follow
    "filename_pattern": r".*\.py",
    "image_scrapers": ("pypte_docs.svg_scraper",),
    "within_subsection_order": "FileNameSortKey",
    "abort_on_example_error": True,
    "download_all_examples": False,
    "remove_config_comments": True,
}

html_theme = "pydata_sphinx_theme"
html_title = f"pyPTE {version}"
html_theme_options = {
    "github_url": "https://github.com/patrk/pyPTE",
    "navbar_align": "left",
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}
html_context = {"default_mode": "light"}
