"""Shared plotting helpers so every example renders connectivity the same way."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# dPTE is centred on 0.5: below means net inflow, above means net outflow
DIVERGING = "RdBu_r"
SEQUENTIAL = "viridis"


def use_house_style() -> None:
    """Apply a consistent, readable style across every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "image.origin": "upper",
        }
    )


def plot_matrix(
    ax: Any,
    matrix: npt.NDArray,
    *,
    title: str = "",
    labels: Sequence[str] | None = None,
    cmap: str = DIVERGING,
    vmin: float | None = None,
    vmax: float | None = None,
    center_on_half: bool = False,
    colorbar: bool = True,
    mask_diagonal: bool = False,
) -> Any:
    """Draw a connectivity matrix with row = source and column = target.

    Uses origin="upper" so row 0 sits at the top, matching how the matrix is
    printed and indexed. Plotting libraries default to the opposite, which
    silently flips a connectivity matrix top to bottom.

    Parameters
    ----------
    center_on_half : bool
        Symmetrise the colour limits around 0.5, the dPTE no-preference point,
        so colour reads as direction rather than magnitude.
    """
    if mask_diagonal:
        # self-transfer is identically zero and would otherwise anchor the
        # colour scale at a value that carries no information
        matrix = matrix.astype(float).copy()
        np.fill_diagonal(matrix, np.nan)

    if center_on_half:
        reach = np.nanmax(np.abs(matrix - 0.5)) or 0.5
        vmin, vmax = 0.5 - reach, 0.5 + reach

    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.grid(False)

    if labels is not None and len(labels) <= 30:
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return image


def save(fig: Any, path: Any) -> None:
    """Write a figure and report where it landed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path}")
