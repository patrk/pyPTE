"""Helpers for the documentation build, importable so Sphinx can pickle them."""

from sphinx_gallery.scrapers import matplotlib_scraper


def svg_scraper(block, block_vars, gallery_conf, **kwargs):
    """Capture figures as SVG so they stay sharp at any zoom."""
    return matplotlib_scraper(block, block_vars, gallery_conf, format="svg", **kwargs)
