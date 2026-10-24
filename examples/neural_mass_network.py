"""
How much recording do you need, and what does more of it buy?
=============================================================

Jansen-Rit columns are much harder than phase oscillators: each column's alpha
rhythm is driven mostly by its own dynamics, so a neighbour's influence is a
small perturbation on top.

Eight columns in a directed ring, recorded for 15 to 200 s, band-passed to the
alpha band and tested against 199 surrogates, over five network realisations.
Each realisation goes from finding nothing to finding every edge, the right
way round, somewhere between 15 and 60 s. The indirect paths a -> b -> c and
a -> b -> c -> d become significant with them: information really does flow
that way. At 15 s some paths are still missing and precision is 0.5; by the
time every edge is found it is a third, and stays there. No recording length
separates the edges from the paths.

Scoring uses ``result.directed``, which marks each significant pair once, from
driver to target. ``result.significant`` marks both orientations of a pair, so
scoring it against a directed ground truth would count the mirror of every
real edge as a false one.

Run with::

    uv run python -m examples.neural_mass_network [--quick]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.signal import butter, sosfiltfilt, welch

from examples.models import kuramoto, neural_mass_model
from examples.utils import style
from examples.utils.plotting import FIGURES, plot_matrix, save, use_house_style
from pyPTE import surrogate_test

FS = 250.0
COUPLING_STRENGTH = 30.0
DELAY = 0.020  # about a fifth of an alpha cycle, where direction is clearest
N_COLUMNS = 8
ALPHA = butter(4, [8.0, 13.0], btype="band", fs=FS, output="sos")


def precision(found: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Share of findings that are edges; undefined where nothing was found."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(found + flagged > 0, found / (found + flagged), np.nan)


def main(quick: bool = False) -> None:
    use_house_style()
    n_surrogates = 99 if quick else 199
    seeds = [1, 2] if quick else [1, 2, 3, 4, 5]
    durations = [15, 30, 60] if quick else [15, 30, 60, 120, 200]

    coupling = kuramoto.ring_with_shortcut(N_COLUMNS, strength=COUPLING_STRENGTH)
    truth = coupling > 0
    hops = (np.arange(N_COLUMNS)[None, :] - np.arange(N_COLUMNS)[:, None]) % N_COLUMNS

    # edges found and other pairs flagged, per realisation and length
    found = np.zeros((len(seeds), len(durations)))
    flagged = np.zeros((len(seeds), len(durations)))
    backwards = 0
    for k, seed in enumerate(seeds):
        print(f"realisation {seed}: simulating {durations[-1]}s ...")
        recording = neural_mass_model.simulate(
            coupling, n_samples=int(durations[-1] * FS), delay=DELAY, seed=seed
        )
        if k == 0:
            reference = recording
        for d, seconds in enumerate(durations):
            # filter each length on its own, so no crop sees samples after it,
            # padded by a second: the default few samples leave the filter's
            # transient inside a short crop, where it can fake a direction
            crop = sosfiltfilt(
                ALPHA, recording[:, : int(seconds * FS)], axis=1, padlen=int(FS)
            )
            result = surrogate_test(crop, n_surrogates=n_surrogates, seed=seed)
            found[k, d] = (result.directed & truth).sum()
            flagged[k, d] = (result.directed & ~truth).sum()
            backwards += int((result.directed.T & truth).sum())
            print(
                f"  {seconds:>4}s  edges found {found[k, d]:.0f} of {truth.sum()}, "
                f"other pairs flagged {flagged[k, d]:.0f}"
            )
        if k == 0:
            first = result.directed

    freqs, power = welch(reference[0], fs=FS, nperseg=1024)
    peak = float(freqs[np.argmax(power)])
    alpha_fraction = float(power[(freqs >= 8) & (freqs < 13)].sum() / power.sum())

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)

    window = slice(0, int(3 * FS))
    for column in range(3):
        axes[0, 0].plot(
            np.arange(int(3 * FS)) / FS,
            reference[column, window],
            lw=0.9,
            label=f"column {column}",
        )
    axes[0, 0].set(
        xlabel="time (s)",
        ylabel="pyramidal potential (mV)",
        title="Jansen-Rit columns produce an alpha rhythm",
    )
    axes[0, 0].legend(fontsize=7, ncols=3)

    axes[0, 1].semilogy(freqs, power, lw=1.2, color=style.BLUE)
    axes[0, 1].axvspan(8, 13, color=style.OBS, alpha=0.15)
    axes[0, 1].set(
        xlim=(0, 40),
        xlabel="frequency (Hz)",
        ylabel="power",
        title=f"Spectrum: peak {peak:.1f} Hz",
    )

    recall = found / truth.sum()
    # pooled over realisations, so ones that found nothing do not count as 0
    pooled = precision(found.sum(axis=0), flagged.sum(axis=0))
    for each, overall, name, color, marker in [
        (recall, recall.mean(axis=0), "recall", style.BLUE, "o"),
        (precision(found, flagged), pooled, "precision", style.OBS, "s"),
    ]:
        axes[1, 0].plot(durations, each.T, color=color, lw=0.6, alpha=0.35)
        axes[1, 0].plot(
            durations,
            overall,
            marker + "-",
            color=color,
            label=f"{name}, over {len(seeds)} realisations",
        )
    axes[1, 0].set(
        xlabel="recording length (s)",
        ylabel="score",
        ylim=(-0.05, 1.05),
        title="Edges and indirect paths become significant together",
    )
    axes[1, 0].legend(fontsize=8, loc="center right")

    # 0 nothing, 1 indirect path flagged, 2 edge missed, 3 edge found
    outcome = 2 * truth.astype(int) + first.astype(int)
    categories = ListedColormap([style.TINT, style.OBS, style.AMBER, style.BLUE])
    plot_matrix(
        axes[1, 1],
        outcome.astype(float),
        title=f"Realisation {seeds[0]} at {durations[-1]}s: directed findings",
        cmap=categories,
        vmin=-0.5,
        vmax=3.5,
        colorbar=False,
        mask_diagonal=True,
    )
    axes[1, 1].legend(
        handles=[
            Patch(color=style.BLUE, label="edge, found"),
            Patch(color=style.AMBER, label="edge, missed"),
            Patch(color=style.OBS, label="no edge, flagged"),
            Patch(color=style.TINT, label="no edge, not flagged"),
        ],
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    save(fig, FIGURES / "neural_mass_network.png")

    print(f"\nover {len(seeds)} realisations")
    print(f"  {'seconds':>8} {'recall':>8} {'precision':>10}")
    for seconds, r, p in zip(durations, recall.mean(axis=0), pooled, strict=True):
        print(f"  {seconds:>8} {r:>8.2f} {p:>10.2f}")
    paths = first & ~truth
    print(
        f"\nrealisation {seeds[0]} at {durations[-1]}s flags "
        f"{int((paths & (hops == 2)).sum())} two-hop and "
        f"{int((paths & (hops == 3)).sum())} three-hop paths"
    )

    assert 8.0 <= peak <= 13.0, (
        f"columns should oscillate in the alpha band, got {peak}"
    )
    assert alpha_fraction > 0.5, "most of the power should sit in the alpha band"
    assert recall[:, -1].mean() >= 0.8, "a long recording should find nearly every edge"
    assert backwards == 0, "no edge should be reported the wrong way round"
    assert pooled[-1] < 0.6, "indirect paths should be flagged along with the edges"

    print(
        f"\nverified: columns oscillate at {peak:.1f} Hz with "
        f"{alpha_fraction:.0%} of power in the alpha band; every edge found"
        "\n          the right way round, but indirect paths came with them."
        "\n\n          A significant dPTE identifies a directed path, not an edge."
        "\n          More data does not separate the two."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer surrogates")
    main(**vars(parser.parse_args()))
