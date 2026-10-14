"""Figures for the "How phase transfer entropy works" tutorial.

The tutorial page shows the analysis; the drawing lives here so that each step
reads as a few lines of pyPTE rather than a screen of matplotlib.
"""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.stats import gaussian_kde

from examples.utils import style


def degrees(angle: float) -> str:
    return f"{np.degrees(angle):+.0f}°".replace("+0°", "0°")


def entropy(counts: npt.ArrayLike) -> float:
    """Shannon entropy in bits of a histogram given as counts or probabilities."""
    p = np.asarray(counts, float).ravel()
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def rose_pair(anywhere: npt.NDArray, after: npt.NDArray, held: int, delay: int) -> Any:
    """Two circular histograms: a phase at any moment, and after a known bin."""
    n_bins = anywhere.size
    width = 2 * np.pi / n_bins
    centres = -np.pi + (np.arange(n_bins) + 0.5) * width
    fig = plt.figure(figsize=(style.WIDTH, 3.4))
    panels = [
        (anywhere, "$Y$ at any moment"),
        (after, f"$Y$ {delay} samples after the grey bin"),
    ]
    for i, (p, title) in enumerate(panels):
        ax = fig.add_axes((0.07 + i * 0.5, 0.12, 0.34, 0.66), projection="polar")
        if i == 1:
            ax.bar(centres[held], 0.6, width=width, color=style.TINT, zorder=0)
        ax.bar(
            centres,
            p,
            width=width * 0.92,
            color=style.BLUE,
            alpha=0.85,
            edgecolor=style.SURFACE,
            linewidth=0.8,
        )
        ax.set_ylim(0, 0.62)
        ax.set_yticks([0.2, 0.4, 0.6], ["0.2", "0.4", "0.6"], fontsize=6.5)
        ax.set_rlabel_position(100)
        ax.set_xticks(np.radians([0, 90, 180, 270]), ["0°", "90°", "±180°", "−90°"])
        ax.grid(color=style.TINT, linewidth=0.8)
        ax.spines["polar"].set_color(style.EDGE)
        style.panel(ax, "ab"[i], title)
        ax.text(
            0.5,
            -0.14,
            f"$H$ = {entropy(p):.2f} bits",
            transform=ax.transAxes,
            ha="center",
            fontsize=8.5,
        )
    return fig


def _ridges(ax, source_now, target_now, target_future_phase, held, n_bins, colour):
    width = 2 * np.pi / n_bins
    future = target_future_phase[target_now == held]
    centre = np.angle(np.exp(1j * future).mean())
    relative = np.angle(np.exp(1j * (future - centre)))
    grid = np.linspace(-np.pi / 2, np.pi / 2, 300)
    outline = gaussian_kde(relative, bw_method=0.18)(grid)
    scale = 0.9 / outline.max()
    sources = source_now[target_now == held]
    keep = [k for k in range(n_bins) if (sources == k).sum() >= 200]
    for row, k in enumerate(keep):
        base = len(keep) - 1 - row
        density = gaussian_kde(relative[sources == k], bw_method=0.18)(grid) * scale
        z = n_bins - base
        ax.fill_between(
            grid, base, base + density, color=colour, alpha=0.85, linewidth=0, zorder=z
        )
        ax.plot(
            grid,
            base + outline * scale,
            color=style.INK,
            lw=0.6,
            ls=(0, (2, 1.5)),
            zorder=z + 0.5,
        )
        ax.plot(grid, base + density, color=style.SURFACE, lw=0.8, zorder=z + 0.4)
    labels = [degrees(-np.pi + (k + 0.5) * width) for k in keep]
    ax.set_yticks(range(len(keep))[::-1], labels)
    ax.set_ylim(-0.3, len(keep) + 0.4)
    ax.set_xlim(-np.pi / 2, np.pi / 2)
    ax.set_xticks(
        np.radians([-90, -45, 0, 45, 90]), ["−90°", "−45°", "0°", "+45°", "+90°"]
    )
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def ridgelines(phase: npt.NDArray, codes: npt.NDArray, delay: int, held: int) -> Any:
    """Each channel's future phase, split by the other channel's present bin."""
    n_bins = int(codes.max()) + 1
    x, y = codes
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(style.WIDTH, 4.4),
        gridspec_kw=dict(wspace=0.35, left=0.12, right=0.98, top=0.9, bottom=0.17),
    )
    _ridges(axes[0], x[:-delay], y[:-delay], phase[1, delay:], held, n_bins, style.BLUE)
    _ridges(axes[1], y[:-delay], x[:-delay], phase[0, delay:], held, n_bins, style.NULL)
    for ax, (source, target), letter in zip(axes, ["XY", "YX"], "ab", strict=True):
        style.panel(ax, letter, f"Does ${source}$ help predict ${target}$?")
        ax.set_ylabel(f"${source}$ phase now (bin centre)")
        ax.set_xlabel(
            f"${target}$ phase {delay} samples later,\n"
            "relative to where it usually lands"
        )
    return fig


def entropy_bars(rows: Sequence[tuple[str, float, float]]) -> Any:
    """Conditional entropies per direction, their gap (PTE) and the dPTE split."""
    fig, (ax, bx) = plt.subplots(
        2,
        1,
        figsize=(style.WIDTH, 3.4),
        gridspec_kw=dict(
            height_ratios=[2.2, 1],
            hspace=1.0,
            left=0.14,
            right=0.96,
            top=0.86,
            bottom=0.13,
        ),
    )
    for i, (_, own, both) in enumerate(rows):
        level = 1 - i
        ax.barh(level + 0.17, own, height=0.3, color=style.NULL)
        ax.barh(level - 0.17, both, height=0.3, color=style.BLUE)
        ax.annotate(
            "",
            xy=(both - 0.004, level + 0.38),
            xytext=(own + 0.004, level + 0.38),
            arrowprops=dict(
                arrowstyle="<->", lw=0.9, color=style.OBS, shrinkA=0, shrinkB=0
            ),
        )
        ax.text(
            own + 0.03,
            level + 0.38,
            f"PTE = {own - both:.3f} bits",
            color=style.OBS,
            fontsize=8,
            va="center",
        )
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("uncertainty of the target's next phase, bits")
    swatch = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (style.NULL, style.BLUE)]
    ax.legend(
        handles=swatch,
        labels=["knowing its own past", "also knowing the other channel"],
        loc="lower left",
        ncol=2,
        bbox_to_anchor=(0, 1.02),
    )
    share = (rows[0][1] - rows[0][2]) / sum(own - both for _, own, both in rows)
    bx.barh(0, share, color=style.OBS, height=0.5)
    bx.barh(0, 1 - share, left=share, color=style.NULL, height=0.5)
    bx.axvline(0.5, color=style.INK, lw=0.8, ls=(0, (2, 1.5)))
    bx.text(
        share / 2,
        0,
        f"dPTE = {share:.2f}",
        color=style.SURFACE,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
    )
    bx.set_xlim(0, 1)
    bx.set_yticks([])
    bx.spines["left"].set_visible(False)
    bx.set_xlabel(r"share of the two transfers running $X \rightarrow Y$")
    style.panel(ax, "a")
    style.panel(bx, "b")
    return fig


def null_densities(results: Sequence[Any], titles: Sequence[str]) -> Any:
    """Surrogate null of dPTE[0, 1] for each result, with the observed value."""
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(style.WIDTH, 2.9),
        gridspec_kw=dict(wspace=0.25, left=0.04, right=0.98, top=0.84, bottom=0.18),
    )
    for ax, result, letter, title in zip(axes, results, "ab", titles, strict=True):
        null = result.null_distribution[:, 0, 1]
        observed, p = result.dPTE[0, 1], result.p_values[0, 1]
        grid = np.linspace(0.3, 0.9, 400)
        density = gaussian_kde(null)(grid)
        ax.fill_between(grid, density, color=style.NULL, alpha=0.7, linewidth=0)
        ax.plot(grid, density, color=style.QUIET, lw=0.8)
        ax.axvline(0.5, color=style.INK, lw=0.7, ls=(0, (1, 2)))
        ax.axvline(observed, color=style.OBS, lw=1.6)
        ax.text(
            observed,
            density.max() * 1.08,
            f" observed {observed:.2f}\n p = {p:.3f}",
            color=style.OBS,
            fontsize=8,
            va="bottom",
        )
        ax.text(
            null.mean() - 0.035,
            density.max() * 0.6,
            "null from\nshifted timing",
            color=style.QUIET,
            fontsize=7.5,
            ha="right",
        )
        ax.set_ylim(0, density.max() * 1.45)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("dPTE")
        style.panel(ax, letter, title)
    return fig


def violins(groups: dict[str, npt.NDArray], highlight: str) -> Any:
    """One violin per condition with every observation drawn on top."""
    fig, ax = plt.subplots(
        figsize=(style.WIDTH, 3.0),
        gridspec_kw=dict(left=0.1, right=0.98, top=0.95, bottom=0.2),
    )
    jitter = np.random.default_rng(0)
    for i, (label, values) in enumerate(groups.items()):
        lead = label == highlight
        body = ax.violinplot(values, positions=[i], widths=0.7, showextrema=False)
        patch = body["bodies"][0]
        patch.set_facecolor(style.BLUE if lead else style.NULL)
        patch.set_alpha(0.35)
        patch.set_edgecolor("none")
        ax.scatter(
            i + jitter.uniform(-0.12, 0.12, values.size),
            values,
            s=9,
            color=style.BLUE if lead else style.QUIET,
            edgecolor=style.SURFACE,
            linewidth=0.4,
            zorder=3,
        )
        median = float(np.median(values))
        ax.plot([i - 0.2, i + 0.2], [median, median], color=style.INK, lw=1.8, zorder=4)
        ax.text(i + 0.38, median, f"{median:.2f}", va="center", fontsize=8)
    ax.axhline(0.5, color=style.INK, lw=0.7, ls=(0, (1, 2)))
    ax.set_xticks(range(len(groups)), list(groups))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(groups) - 0.2)
    return fig, ax


def _network(ax, positions, edges, colours, widths):
    for (i, j), colour, lw in zip(edges, colours, widths, strict=True):
        ax.add_patch(
            FancyArrowPatch(
                positions[i],
                positions[j],
                arrowstyle="-|>,head_length=4,head_width=2.4",
                connectionstyle="arc3,rad=0.18",
                shrinkA=11,
                shrinkB=11,
                lw=lw,
                color=colour,
                zorder=2,
            )
        )
    for k, (px, py) in enumerate(positions):
        ax.add_patch(
            Circle(
                (px, py),
                0.13,
                facecolor=style.SURFACE,
                edgecolor=style.INK,
                lw=1,
                zorder=3,
            )
        )
        ax.text(px, py, str(k), ha="center", va="center", fontsize=8.5, zorder=4)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def network_pair(truth: npt.NDArray, result: Any) -> Any:
    """Ground-truth ring beside the significant dPTE edges, one arrow per pair."""
    m = truth.shape[0]
    angles = np.pi / 2 - 2 * np.pi * np.arange(m) / m
    positions = np.c_[np.cos(angles), np.sin(angles)]
    true_edges = [(i, j) for i in range(m) for j in range(m) if truth[i, j]]
    found = [
        (i, j)
        for i in range(m)
        for j in range(m)
        if result.significant[i, j] and result.dPTE[i, j] > 0.5
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(style.WIDTH, 3.6),
        gridspec_kw=dict(wspace=0.1, left=0.02, right=0.98, top=0.86, bottom=0.06),
    )
    _network(
        axes[0],
        positions,
        true_edges,
        [style.INK] * len(true_edges),
        [1.4] * len(true_edges),
    )
    _network(
        axes[1],
        positions,
        found,
        [style.OBS if truth[e] else style.EDGE for e in found],
        [0.6 + 10 * (result.dPTE[e] - 0.5) for e in found],
    )
    real = sum(bool(truth[e]) for e in found)
    style.panel(axes[0], "a", "Ground truth: each node drives the next")
    style.panel(
        axes[1], "b", f"Recovered: {len(found)} significant arrows, {real} real"
    )
    lines = [plt.Line2D([], [], color=c, lw=2) for c in (style.OBS, style.EDGE)]
    axes[1].legend(
        handles=lines,
        labels=["true edge", "indirect path"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
    )
    return fig


def _chord(ax, positions, order, left, labels, edges, colour_of):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    for i, j in edges:
        start, end = positions[i] * 0.97, positions[j] * 0.97
        curve = Path(
            [start, (start + end) * 0.15, end], [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        )
        ax.add_patch(PathPatch(curve, fc="none", ec=colour_of(i, j), lw=0.7, alpha=0.8))
        ax.plot(*end, marker="o", ms=2, color=colour_of(i, j))
    for region in order:
        angle = np.arctan2(positions[region][1], positions[region][0])
        ax.add_patch(
            Circle(
                positions[region],
                0.018,
                zorder=3,
                color=style.INK if region in left else style.QUIET,
            )
        )
        rotation = np.degrees(angle)
        flip = 90 < rotation % 360 < 270
        ax.text(
            *(positions[region] * 1.06),
            labels[region],
            fontsize=4.2,
            rotation=rotation + (180 if flip else 0),
            ha="right" if flip else "left",
            va="center",
            rotation_mode="anchor",
            color=style.QUIET,
        )
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")


def chord_pair(
    weights: npt.NDArray, dPTE: npt.NDArray, labels: Sequence[str], n_edges: int = 60
) -> Any:
    """Strongest one-way tracts beside the strongest dPTE flows, regions on a circle."""
    n = weights.shape[0]
    left = {i for i, label in enumerate(labels) if label.lower().startswith("l")}
    right = [i for i in range(n) if i not in left]
    # right hemisphere down the right side, left hemisphere up the left
    order = right + sorted(left, reverse=True)
    angles = {r: np.pi / 2 - 2 * np.pi * (k + 0.5) / n for k, r in enumerate(order)}
    positions = {r: np.array([np.cos(a), np.sin(a)]) for r, a in angles.items()}
    one_way = (weights > 0) & (weights.T == 0)
    tracts = sorted(
        ((weights[i, j], i, j) for i in range(n) for j in range(n) if one_way[i, j]),
        reverse=True,
    )[:n_edges]
    flows = sorted(
        (
            (dPTE[i, j], i, j)
            for i in range(n)
            for j in range(n)
            if i != j and dPTE[i, j] > 0.5
        ),
        reverse=True,
    )[:n_edges]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(style.WIDTH, 4.0),
        gridspec_kw=dict(wspace=0.05, left=0.01, right=0.99, top=0.88, bottom=0.06),
    )
    _chord(
        axes[0],
        positions,
        order,
        left,
        labels,
        [(i, j) for _, i, j in tracts],
        lambda i, j: style.INK,
    )
    _chord(
        axes[1],
        positions,
        order,
        left,
        labels,
        [(i, j) for _, i, j in flows],
        lambda i, j: style.OBS if one_way[i, j] else style.EDGE,
    )
    hits = sum(bool(one_way[i, j]) for _, i, j in flows)
    style.panel(axes[0], "a", f"Anatomy: {n_edges} strongest one-way tracts")
    style.panel(
        axes[1], "b", f"dPTE: {n_edges} strongest flows, {hits} on a one-way tract"
    )
    fig.text(
        0.5,
        0.02,
        "Left hemisphere on the left (dark nodes), right hemisphere on "
        "the right. A dot marks the target end.",
        ha="center",
        fontsize=7,
        color=style.QUIET,
    )
    return fig
