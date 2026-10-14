"""
How phase transfer entropy works
================================

One pair of oscillators, X driving Y, followed from raw phase to a
connectivity graph. Every number and figure on this page is computed when the
documentation is built.

.. math::

   \\mathrm{PTE}(X \\to Y) = H(Y_{t+\\delta} \\mid Y_t)
   - H(Y_{t+\\delta} \\mid Y_t, X_t)

In words: how much less uncertain Y's next phase becomes once you also know
X's current phase. The steps below build each piece of that line.
"""

# %%
# The example pair
# ----------------
# Two delayed Kuramoto oscillators where X drives Y, and nothing flows back.
# Steps 1 to 3 use a long run so the histograms are smooth; steps 4 and 5 use
# recordings of realistic length. Phase is cut into 12 equal bins throughout,
# so the probabilities drawn are exactly the ones pyPTE counts.

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.stats import gaussian_kde

from examples.models import kuramoto
from examples.utils import style
from pyPTE import PTE, surrogate_test
from pyPTE.core.pyPTE import get_delay, get_phase

style.use()
K = 12
PAIR = dict(noise=1.0, freq_spread=0.5)
DRIVE = np.array([[0, 4.0], [0, 0]])

signal = kuramoto.simulate(DRIVE, n_samples=300_000, seed=2, **PAIR)
phase = get_phase(signal)
delay = get_delay(phase)
codes = np.clip(np.floor((phase + np.pi) / (2 * np.pi / K)), 0, K - 1).astype(int)
dPTE, raw_PTE = PTE(signal, binning=K, delay=delay)

x, y = codes
y_future, y_now, x_now, x_future = y[delay:], y[:-delay], x[:-delay], x[delay:]
width = 2 * np.pi / K


def entropy(counts):
    p = np.asarray(counts, float).ravel()
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def degrees(angle):
    return f"{np.degrees(angle):+.0f}°".replace("+0°", "0°")


print(
    f"delay {delay} samples, PTE X→Y {raw_PTE[0, 1]:.3f} bits, "
    f"Y→X {raw_PTE[1, 0]:.3f} bits, dPTE {dPTE[0, 1]:.2f}"
)

# %%
# 1. Entropy measures how spread out a phase is
# ---------------------------------------------
# **Entropy** :math:`H` is the average number of yes/no questions needed to
# pin down which bin a value falls in: largest for a flat histogram, zero for
# a single spike. **Conditional entropy** :math:`H(A \mid B)` is the same once
# :math:`B` is known.
#
# Over the whole recording Y visits every bin equally (**a**). But 12 samples
# after Y sat in one bin, it has almost always moved half a cycle on (**b**):
# knowing where Y was cuts its uncertainty sharply.

held = int(np.bincount(y_now).argmax())
anywhere = np.bincount(y, minlength=K) / y.size
after = np.bincount(y_future[y_now == held], minlength=K)
after = after / after.sum()
centres = -np.pi + (np.arange(K) + 0.5) * width

fig = plt.figure(figsize=(style.WIDTH, 3.4))
for i, (p, title) in enumerate(
    [
        (anywhere, "$Y$ at any moment"),
        (after, f"$Y$ {delay} samples after the grey bin"),
    ]
):
    ax = fig.add_axes([0.07 + i * 0.5, 0.12, 0.34, 0.66], projection="polar")
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
    ax.set_yticks(
        [0.2, 0.4, 0.6], ["0.2", "0.4", "0.6"], fontsize=6.5, color=style.QUIET
    )
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

# %%
# 2. Transfer means X sharpens Y's future even further
# ----------------------------------------------------
# Hold Y's current bin fixed and split its future by X's current phase. Each
# ridge in **a** is Y's future for one bin of X; the dashed outline ignores X.
# The peaks move away from the outline, so each is narrower: X carries
# information about where Y goes. In **b** the roles are swapped and every
# ridge sits on its outline. Bins of the other channel that hardly occur
# alongside the held bin are left out.
#
# **Transfer entropy** is that drop in conditional entropy, the conditional
# mutual information :math:`I(Y_{t+\delta} ; X_t \mid Y_t)`. **Phase** transfer
# entropy computes it on Hilbert phases, so amplitude plays no part.


def ridges(ax, source_now, target_now, target_future_phase, colour):
    selected = target_now == held
    future = target_future_phase[selected]
    centre = np.angle(np.exp(1j * future).mean())
    relative = np.angle(np.exp(1j * (future - centre)))
    grid = np.linspace(-np.pi / 2, np.pi / 2, 300)
    outline = gaussian_kde(relative, bw_method=0.18)(grid)
    scale = 0.9 / outline.max()
    sources = source_now[selected]
    keep = [k for k in range(K) if (sources == k).sum() >= 200]
    for row, k in enumerate(keep):
        base = len(keep) - 1 - row
        density = gaussian_kde(relative[sources == k], bw_method=0.18)(grid) * scale
        ax.fill_between(
            grid,
            base,
            base + density,
            color=colour,
            alpha=0.85,
            linewidth=0,
            zorder=K - base,
        )
        ax.plot(
            grid,
            base + outline * scale,
            color=style.INK,
            lw=0.6,
            ls=(0, (2, 1.5)),
            zorder=K - base + 0.5,
        )
        ax.plot(
            grid, base + density, color=style.SURFACE, lw=0.8, zorder=K - base + 0.4
        )
    ax.set_yticks(
        range(len(keep))[::-1], [degrees(-np.pi + (k + 0.5) * width) for k in keep]
    )
    ax.set_ylim(-0.3, len(keep) + 0.4)
    ax.set_xlim(-np.pi / 2, np.pi / 2)
    ax.set_xticks(
        np.radians([-90, -45, 0, 45, 90]), ["−90°", "−45°", "0°", "+45°", "+90°"]
    )
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(style.WIDTH, 4.4),
    gridspec_kw=dict(wspace=0.35, left=0.12, right=0.98, top=0.9, bottom=0.17),
)
ridges(axes[0], x_now, y_now, phase[1, delay:], style.BLUE)
ridges(axes[1], y_now, x[:-delay], phase[0, delay:], style.NULL)
for ax, (source, target), letter in zip(axes, ["XY", "YX"], "ab", strict=True):
    style.panel(ax, letter, f"Does ${source}$ help predict ${target}$?")
    ax.set_ylabel(f"${source}$ phase now (bin centre)")
    ax.set_xlabel(
        f"${target}$ phase {delay} samples later,\nrelative to where it usually lands"
    )

# %%
# 3. One number per direction, and their ratio
# --------------------------------------------
# Each pair of bars in **a** is one direction: the uncertainty of the
# target's future given its own past, then given the other channel as well.
# The gap is PTE. **dPTE** in **b** is X→Y's share of the two transfers,
# :math:`\mathrm{PTE}(X\to Y) / (\mathrm{PTE}(X\to Y) + \mathrm{PTE}(Y\to X))`.
# It answers "which way?", not "how strong?", and 0.5 means no net direction.


def encode(*arrays):
    code = np.zeros_like(arrays[0])
    for a in arrays:
        code = code * K + a
    return code


def conditional_entropy(future, *given):
    return entropy(np.bincount(encode(future, *given))) - entropy(
        np.bincount(encode(*given))
    )


rows = [
    (
        "$X$ → $Y$",
        conditional_entropy(y_future, y_now),
        conditional_entropy(y_future, y_now, x_now),
    ),
    (
        "$Y$ → $X$",
        conditional_entropy(x_future, x[:-delay]),
        conditional_entropy(x_future, x[:-delay], y_now),
    ),
]
fig, (ax, bx) = plt.subplots(
    2,
    1,
    figsize=(style.WIDTH, 3.4),
    gridspec_kw=dict(
        height_ratios=[2.2, 1], hspace=1.0, left=0.14, right=0.96, top=0.86, bottom=0.13
    ),
)
for i, (_name, own, both) in enumerate(rows):
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
ax.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color=style.NULL),
        plt.Rectangle((0, 0), 1, 1, color=style.BLUE),
    ],
    labels=["knowing its own past", "also knowing the other channel"],
    loc="lower left",
    ncol=2,
    bbox_to_anchor=(0, 1.02),
)
share = (rows[0][1] - rows[0][2]) / sum(r[1] - r[2] for r in rows)
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
bx.set_xlabel("share of the two transfers running $X$ → $Y$")
style.panel(ax, "a")
style.panel(bx, "b")

# %%
# 4. A value is only meaningful against its own null
# --------------------------------------------------
# A **surrogate test** rebuilds the statistic many times on data where the
# effect cannot exist, here by shifting each channel's timing at random, which
# keeps each signal intact but breaks their alignment. The p-value is the
# share of surrogates at least as extreme as the real value.
#
# The coupled pair lands far outside its null (**a**). Two independent
# signals with Y much noisier land even further from 0.5, but so does their
# whole null (**b**): the cleaner channel is easier to predict and looks like
# a driver. dPTE alone cannot tell these apart; the null can.

rng = np.random.default_rng(4)
coupled = kuramoto.simulate(DRIVE, n_samples=12_000, seed=3, **PAIR)
coupled = coupled + 0.2 * rng.standard_normal(coupled.shape)
unrelated = kuramoto.simulate(np.zeros((2, 2)), n_samples=12_000, seed=5, **PAIR)
unrelated = unrelated + np.array([[0.2], [1.5]]) * rng.standard_normal(unrelated.shape)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(style.WIDTH, 2.9),
    gridspec_kw=dict(wspace=0.25, left=0.04, right=0.98, top=0.84, bottom=0.18),
)
for ax, data, letter, title in zip(
    axes,
    [coupled, unrelated],
    "ab",
    ["$X$ drives $Y$", "Independent, $Y$ much noisier"],
    strict=True,
):
    result = surrogate_test(data, n_surrogates=1000, seed=0, keep_null=True, binning=K)
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

# %%
# 5. Many short recordings beat one long one
# ------------------------------------------
# Each dot is one 12 s epoch. Coupled epochs sit above 0.5 almost every time,
# a consistency that :func:`pyPTE.group_test` detects easily. The noisy-Y
# condition sits higher still with no coupling at all, which is why
# conditions are compared with each other rather than with 0.5.


def epochs(kind, n=80):
    values = []
    for s in range(n):
        if kind == "coupled":
            data = kuramoto.simulate(DRIVE, n_samples=3000, seed=100 + s, **PAIR)
            noise = np.array([[0.2], [0.2]])
        else:
            data = kuramoto.simulate(
                np.zeros((2, 2)), n_samples=3000, seed=300 + s, **PAIR
            )
            noise = (
                np.array([[0.2], [0.2]])
                if kind == "equal"
                else np.array([[0.2], [1.5]])
            )
        data = data + noise * np.random.default_rng(s).standard_normal(data.shape)
        values.append(PTE(data, binning=K)[0][0, 1])
    return np.array(values)


conditions = [
    ("coupled", "$X$ drives $Y$"),
    ("equal", "independent,\nequal noise"),
    ("unequal", "independent,\n$Y$ much noisier"),
]
fig, ax = plt.subplots(
    figsize=(style.WIDTH, 3.0),
    gridspec_kw=dict(left=0.1, right=0.98, top=0.95, bottom=0.2),
)
jitter = np.random.default_rng(0)
for i, (kind, _label) in enumerate(conditions):
    values = epochs(kind)
    colour = style.BLUE if kind == "coupled" else style.QUIET
    body = ax.violinplot(values, positions=[i], widths=0.7, showextrema=False)[
        "bodies"
    ][0]
    body.set_facecolor(style.BLUE if kind == "coupled" else style.NULL)
    body.set_alpha(0.35)
    body.set_edgecolor("none")
    ax.scatter(
        i + jitter.uniform(-0.12, 0.12, values.size),
        values,
        s=9,
        color=colour,
        edgecolor=style.SURFACE,
        linewidth=0.4,
        zorder=3,
    )
    median = np.median(values)
    ax.plot([i - 0.2, i + 0.2], [median, median], color=style.INK, lw=1.8, zorder=4)
    ax.text(i + 0.38, median, f"{median:.2f}", va="center", fontsize=8)
ax.axhline(0.5, color=style.INK, lw=0.7, ls=(0, (1, 2)))
ax.set_xticks(range(3), [c[1] for c in conditions])
ax.tick_params(axis="x", length=0)
ax.set_ylabel("dPTE per 12 s epoch")
ax.set_xlim(-0.6, 2.8)

# %%
# 6. Significant edges make a graph, with predictable errors
# ----------------------------------------------------------
# Eight oscillators in a ring, each driving the next (**a**). Only edges
# surviving the surrogate test with false-discovery-rate control are drawn in
# **b**, width by dPTE − 0.5. Direction is right on every arrow, but some
# arrows skip a node: information really does flow that way, through an
# intermediate, and a pairwise measure cannot tell a direct connection from an
# indirect one.
#
# The **false discovery rate** bounds the expected share of false edges among
# those reported. With 28 channel pairs, :func:`pyPTE.surrogates_needed` says
# how many surrogates an isolated edge needs before it can pass at all.

ring_coupling = kuramoto.ring_with_shortcut(8, strength=3.0)
ring = kuramoto.simulate(ring_coupling, n_samples=12_000, seed=1)
truth = kuramoto.net_flow(ring_coupling)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ring_result = surrogate_test(ring, n_surrogates=1200, seed=0)

m = truth.shape[0]
angles = np.pi / 2 - 2 * np.pi * np.arange(m) / m
positions = np.c_[np.cos(angles), np.sin(angles)]


def network(ax, edges, colours, widths):
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


true_edges = [(i, j) for i in range(m) for j in range(m) if truth[i, j]]
found = [
    (i, j)
    for i in range(m)
    for j in range(m)
    if ring_result.significant[i, j] and ring_result.dPTE[i, j] > 0.5
]
fig, axes = plt.subplots(
    1,
    2,
    figsize=(style.WIDTH, 3.6),
    gridspec_kw=dict(wspace=0.1, left=0.02, right=0.98, top=0.86, bottom=0.06),
)
network(axes[0], true_edges, [style.INK] * len(true_edges), [1.4] * len(true_edges))
network(
    axes[1],
    found,
    [style.OBS if truth[e] else style.EDGE for e in found],
    [0.6 + 10 * (ring_result.dPTE[e] - 0.5) for e in found],
)
style.panel(axes[0], "a", "Ground truth: each node drives the next")
style.panel(
    axes[1],
    "b",
    f"Recovered: {len(found)} significant arrows, "
    f"{sum(bool(truth[e]) for e in found)} of them real",
)
axes[1].legend(
    handles=[
        plt.Line2D([], [], color=style.OBS, lw=2),
        plt.Line2D([], [], color=style.EDGE, lw=2),
    ],
    labels=["true edge", "indirect path"],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=2,
)

# %%
# 7. At brain scale, flow is not anatomy
# --------------------------------------
# The same lesson grows with the network. On a 76-region connectome almost
# every pair of regions is joined by a short indirect route, and the
# strongest dPTE flows rarely coincide with direct anatomical tracts. The
# simulation is too heavy to run on every documentation build; see the
# ``tvb_connectome`` example in the repository.
