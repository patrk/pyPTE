"""
A two-condition analysis on epochs, which is how M/EEG data usually arrive.
===========================================================================

Runs the whole pipeline: band-pass, epoch, dPTE per epoch, group statistics,
correction. Three settings make the answer trustworthy, and the example uses
all of them: time-reversal-differenced dPTE, so that differences in
signal-to-noise ratio cannot pose as direction; bins sized by how tightly the
channels lock (binning="locking"), since too few let the delay decide a
coupling's direction; and epochs long enough to carry
directional information, 8 s here.

Compares both corrections. FDR bounds the share of wrong edges among those
reported; cluster permutation bounds the chance of reporting a spurious
component, which is what M/EEG reviewers usually expect.

The last thing it shows is the catch: dPTE grows with path length, so indirect
pairs outrank the real edges.

Run with::

    uv run python -m examples.epoched_analysis [--quick]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from examples.utils import style
from examples.utils.plotting import FIGURES, plot_matrix, save, use_house_style
from pyPTE import cluster_permutation, group_contrast, group_test
from pyPTE.core.pyPTE import compute_segments, prepare_phase

FS = 250.0
EPOCH_SECONDS = 8.0
LAG = 10  # samples per hop, 40 ms
PAD = 250  # filtered with padding, then cropped, as when epoching filtered data
LABELS = ["a", "b", "c", "d"]
SOS = butter(4, [8.0, 12.0], btype="band", fs=FS, output="sos")


def oscillator(rng: np.random.Generator, n: int, freq: float = 10.0) -> np.ndarray:
    """A drifting 10 Hz rhythm that starts at a random phase.

    Starting every rhythm at the same phase would line unrelated channels up
    identically in every epoch, which a group test reads as direction.
    """
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    start = rng.uniform(0.0, 2 * np.pi)
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift + start)


def epoch(n_samples: int, coupling: float, seed: int) -> np.ndarray:
    """One epoch of a four-channel chain a -> b -> c -> d.

    Each channel takes the previous one's rhythm LAG samples late, mixed with a
    rhythm of its own, and every channel gets the same measurement noise. With
    unequal noise, dPTE would point from the cleaner channel to the noisier
    one whatever the wiring.
    """
    rng = np.random.default_rng(seed)
    n = n_samples + 2 * PAD + 3 * LAG
    sources = [oscillator(rng, n)]
    for _ in range(3):
        late = np.concatenate([np.zeros(LAG), sources[-1][:-LAG]])
        sources.append(coupling * late + (1.0 - coupling) * oscillator(rng, n))
    channels = np.vstack(sources) + 0.5 * rng.standard_normal((4, n))
    window = slice(PAD + 3 * LAG, PAD + 3 * LAG + n_samples)
    return sosfiltfilt(SOS, channels, axis=1)[:, window]


def dpte_per_epoch(n_epochs: int, coupling: float, offset: int = 0) -> np.ndarray:
    """Time-reversal-differenced dPTE per epoch, delay and bins shared."""
    n_samples = int(EPOCH_SECONDS * FS)
    epochs = np.array([epoch(n_samples, coupling, s + offset) for s in range(n_epochs)])
    d_phase, delay, n_bins = prepare_phase(
        epochs.reshape(-1, n_samples), binning="locking", channels_per_group=4
    )
    dPTE, _ = compute_segments(
        list(d_phase.reshape(n_epochs, 4, -1)), delay, n_bins, time_reversal=True
    )
    return dPTE


def truth_matrix() -> np.ndarray:
    truth = np.zeros((4, 4), dtype=bool)
    for i in range(3):
        truth[i, i + 1] = True
    return truth


def main(quick: bool = False) -> None:
    use_house_style()
    n_epochs = 20 if quick else 40
    n_permutations = 200 if quick else 1000
    epoch_counts = [5, 10, 20] if quick else [5, 10, 20, 40]

    truth = truth_matrix()

    print(f"computing dPTE for {n_epochs} epochs per condition ...")
    task = dpte_per_epoch(n_epochs, coupling=0.9)
    rest = dpte_per_epoch(n_epochs, coupling=0.0, offset=5000)

    print("group test (task vs chance) ...")
    against_chance = group_test(task)
    print(" ", against_chance.summary())

    print("paired contrast (task vs rest) ...")
    contrast = group_contrast(task, rest)
    print(" ", contrast.summary())

    print(f"cluster permutation ({n_permutations} permutations) ...")
    clusters = cluster_permutation(task, rest, n_permutations=n_permutations, seed=0)
    print(" ", clusters.summary())

    # -- how many epochs are actually needed? ------------------------------
    print("sweeping epoch count ...")
    detected = []
    for count in epoch_counts:
        result = group_test(task[:count])
        found = int((result.significant & truth).sum())
        detected.append(found / int(truth.sum()))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)

    sample_epoch = epoch(int(EPOCH_SECONDS * FS), 0.9, 0)[:, : int(FS)]
    for channel in range(4):
        axes[0, 0].plot(
            np.arange(sample_epoch.shape[1]) / FS,
            sample_epoch[channel] + 3.0 * channel,
            lw=0.9,
            label=LABELS[channel],
        )
    axes[0, 0].set(
        xlabel="time (s)",
        ylabel="channel (offset)",
        title="First second of an epoch, chain a->b->c->d",
        yticks=[],
    )
    axes[0, 0].legend(fontsize=7, ncols=4)

    # dPTE against how many hops separate the pair, the central caveat
    hops, values = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            hops.append(j - i)
            values.append(float(task[:, i, j].mean()))
    axes[0, 1].scatter(hops, values, s=45, color=style.OBS, zorder=3)
    for distance in (1, 2, 3):
        mean_at = np.mean(
            [v for h, v in zip(hops, values, strict=True) if h == distance]
        )
        axes[0, 1].plot([distance - 0.25, distance + 0.25], [mean_at] * 2, color="0.3")
    axes[0, 1].axhline(0.5, ls="--", lw=1, color="0.5")
    axes[0, 1].set(
        xticks=[1, 2, 3],
        xlabel="hops along the chain",
        ylabel="mean dPTE",
        title="CAVEAT: indirect pairs score highest",
    )
    axes[0, 1].text(
        1.0,
        min(values) - 0.015,
        "only 1 hop is a real edge",
        fontsize=7,
        color="0.35",
        style="italic",
        ha="center",
        va="top",
    )
    plot_matrix(
        axes[0, 2],
        task.mean(axis=0),
        title=f"Mean dPTE over {n_epochs} epochs",
        labels=LABELS,
        center_on_half=True,
        mask_diagonal=True,
    )

    plot_matrix(
        axes[1, 0],
        np.where(contrast.significant, contrast.effect, np.nan),
        title=(
            f"Paired contrast, FDR corrected\n{contrast.n_significant} edges retained"
        ),
        labels=LABELS,
        cmap="RdBu_r",
        vmin=-0.25,
        vmax=0.25,
        mask_diagonal=True,
    )
    axes[1, 0].set_facecolor("#f1f3f5")

    plot_matrix(
        axes[1, 1],
        np.where(clusters.significant, clusters.statistic, np.nan),
        title=(
            "Cluster permutation\n"
            f"{int(clusters.significant.sum())} edges in surviving clusters"
        ),
        labels=LABELS,
        cmap="RdBu_r",
        vmin=-8,
        vmax=8,
        mask_diagonal=True,
    )
    axes[1, 1].set_facecolor("#f1f3f5")

    axes[1, 2].plot(epoch_counts, detected, "o-", color=style.BLUE, lw=1.6)
    axes[1, 2].axhline(1.0, ls="--", lw=1, color="0.5")
    axes[1, 2].set(
        xlabel="epochs used",
        ylabel="fraction of true edges found",
        ylim=(-0.05, 1.1),
        title=f"Epochs needed, at {EPOCH_SECONDS:.0f}s each",
    )

    save(fig, FIGURES / "epoched_analysis.png")

    print("\nresults")
    print(f"  true edges                    {int(truth.sum())}")
    print(
        f"  found by FDR contrast         {int((contrast.significant & truth).sum())}"
    )
    print(
        f"  found inside clusters         {int((clusters.significant & truth).sum())}"
    )
    print(f"  total data per condition      {n_epochs * EPOCH_SECONDS:.0f}s")
    print("\n  epochs -> fraction of true edges found")
    for count, fraction in zip(epoch_counts, detected, strict=True):
        print(f"    {count:>3} epochs ({count * EPOCH_SECONDS:>3.0f}s): {fraction:.2f}")

    direct = float(np.mean([task[:, i, i + 1].mean() for i in range(3)]))
    three_hop = float(task[:, 0, 3].mean())

    print("\n  mean dPTE by separation along the chain")
    for distance in (1, 2, 3):
        pairs = [task[:, i, i + distance].mean() for i in range(4 - distance)]
        marker = "  <- the only real edges" if distance == 1 else ""
        print(f"    {distance} hop(s): {np.mean(pairs):.3f}{marker}")

    # direction is recovered; which edges are direct is not
    forward = [task[:, i, j].mean() for i in range(4) for j in range(i + 1, 4)]
    assert all(value > 0.5 for value in forward), (
        "every downstream pair should show forward flow"
    )
    assert clusters.significant.any(), "the cluster test should find the chain"
    assert detected[-1] >= detected[0], "more epochs should not find fewer edges"
    assert against_chance.n_significant > 0, "task connectivity should beat chance"
    assert three_hop > direct, (
        "the indirect-path caveat should be visible; if this fails, the "
        "demonstration of the limitation is no longer valid"
    )

    print(
        f"\nverified: {n_epochs} epochs of {EPOCH_SECONDS:.0f}s recover the"
        "\n          direction of flow throughout the chain."
        f"\n\n          But the three-hop pair a->d scores {three_hop:.3f} against"
        f"\n          {direct:.3f} for the real one-hop edges: a longer path is a"
        "\n          longer lag, which PTE detects more easily. pyPTE recovers"
        "\n          the flow, not the wiring: ranking pairs by dPTE puts the"
        "\n          connections that do not exist at the top."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer epochs")
    main(**vars(parser.parse_args()))
