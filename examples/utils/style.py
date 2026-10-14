"""House style for pyPTE figures.

Journal conventions, sized for a screen: no titles inside figures (captions
carry the story), bold lowercase panel letters, a sans-serif type scale and
quiet axes. Colours have fixed roles so that every figure reads the same way:

- ``SEQ``: one blue ramp for probability and other magnitudes
- ``DIV``: blue to red through neutral grey for dPTE around 0.5
- ``NULL``: grey for reference distributions and "nothing here"
- ``OBS``: red for the observed value or a true edge
- ``BLUE``: the single accent for data that is not a null
- ``AMBER``: a second category where one accent is not enough
"""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#1a1a1a"
QUIET = "#666666"
EDGE = "#9a9a9a"
TINT = "#f2f2f2"
SURFACE = "#ffffff"
NULL = "#b8b8b8"
OBS = "#d6403f"
BLUE = "#2a78d6"
# a second category beside BLUE, e.g. "symmetric" or "missed"
AMBER = "#c98500"

SEQ = LinearSegmentedColormap.from_list(
    "pypte_seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list(
    "pypte_div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32a2a"]
)

# full docs column width, in inches; heights are chosen per figure
WIDTH = 7.2

RC: dict[str, Any] = {
    # Helvetica on macOS, Liberation Sans (metric-compatible with Arial) on the
    # Linux docs runner, DejaVu Sans as matplotlib's own last resort
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "mathtext.fontset": "custom",
    # maths follows the text face; symbols that face lacks, such as arrows,
    # come from STIX sans rather than a placeholder box
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.fallback": "stixsans",
    "axes.edgecolor": EDGE,
    "axes.labelcolor": INK,
    "xtick.color": QUIET,
    "ytick.color": QUIET,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
}


def use() -> None:
    """Apply the house style to every figure created afterwards."""
    plt.rcParams.update(RC)


def panel(ax: Any, letter: str, title: str = "") -> None:
    """Title an axis with a bold panel letter, journal style."""
    ax.set_title(rf"$\bf{{{letter}}}$   {title}", loc="left", color=INK)
