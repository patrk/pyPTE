"""How much recording do you need, and what does more of it actually buy?

Phase oscillators make directed coupling easy to see. Jansen-Rit columns do
not: they produce a realistic alpha rhythm whose phase is dominated by each
column's own dynamics, so the influence of a neighbour is a small perturbation
on top. This example asks the practical question that follows - how long a
recording is needed before that influence is detectable at all.

The answer has an uncomfortable second half. Recall improves with recording
length and then saturates, but precision *falls*, because a bivariate measure
cannot tell a direct connection from a transitive one. Given enough data the
indirect path a -> b -> c becomes detectable as a -> c, and it is not a
statistical error: information genuinely does flow that way. It simply is not
an edge of the network you were trying to recover.

So there is an optimum recording length for network reconstruction, which is
not the intuition most people bring to this.

Run with:
    uv run python -m examples.neural_mass_network [--quick]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.signal import welch

from examples.models import kuramoto, neural_mass_model
from examples.utils.plotting import plot_matrix, save, use_house_style
from pyPTE import surrogate_test

FS = 250.0
COUPLING_STRENGTH = 30.0
DELAY = 0.020  # about a fifth of an alpha cycle, where direction is clearest
N_COLUMNS = 5
FIGURES = Path(__file__).resolve().parent / "figures"


def score(result, truth: np.ndarray) -> dict:
    """Precision and recall of the significant edges against ground truth."""
    off_diagonal = ~np.eye(truth.shape[0], dtype=bool)
    found = result.significant & off_diagonal
    true_edges = truth & off_diagonal

    true_positives = int((found & true_edges).sum())
    false_positives = int((found & ~true_edges).sum())

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "recall": true_positives / max(int(true_edges.sum()), 1),
        "precision": true_positives / max(true_positives + false_positives, 1),
        "significant": found,
    }


def main(quick: bool = False) -> None:
    use_house_style()
    n_surrogates = 40 if quick else 100
    durations = [30, 80, 160] if quick else [30, 60, 90, 120, 160, 200]

    coupling = kuramoto.ring_with_shortcut(N_COLUMNS, strength=COUPLING_STRENGTH)
    truth = coupling > 0

    # -- confirm the columns really are producing an alpha rhythm ----------
    reference = neural_mass_model.simulate(
        coupling, n_samples=int(60 * FS), delay=DELAY, seed=1
    )
    freqs, power = welch(reference[0], fs=FS, nperseg=1024)
    peak = float(freqs[np.argmax(power)])
    alpha_fraction = float(power[(freqs >= 8) & (freqs < 13)].sum() / power.sum())
    print(f"alpha check: peak {peak:.2f} Hz, {alpha_fraction:.0%} of power in 8-13 Hz")

    # -- how does detection change with recording length? ------------------
    rows = []
    for seconds in durations:
        print(f"simulating and testing {seconds:>4}s of recording ...")
        signal = neural_mass_model.simulate(
            coupling, n_samples=int(seconds * FS), delay=DELAY, seed=1
        )
        result = surrogate_test(signal, n_surrogates=n_surrogates, seed=0)
        rows.append({"seconds": seconds, "result": result, **score(result, truth)})

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)

    window = slice(0, int(3 * FS))
    for column in range(min(3, N_COLUMNS)):
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

    axes[0, 1].semilogy(freqs, power, lw=1.2, color="#2f6f9f")
    axes[0, 1].axvspan(8, 13, color="#c2453f", alpha=0.15)
    axes[0, 1].set(
        xlim=(0, 40),
        xlabel="frequency (Hz)",
        ylabel="power",
        title=f"Spectrum: peak {peak:.1f} Hz",
    )

    seconds = [row["seconds"] for row in rows]
    axes[1, 0].plot(
        seconds,
        [row["recall"] for row in rows],
        "o-",
        color="#2f6f9f",
        label="recall (edges found)",
    )
    axes[1, 0].plot(
        seconds,
        [row["precision"] for row in rows],
        "s-",
        color="#c2453f",
        label="precision (of those, correct)",
    )
    axes[1, 0].set(
        xlabel="recording length (s)",
        ylabel="score",
        ylim=(-0.05, 1.05),
        title="More data finds every edge, then adds wrong ones",
    )
    axes[1, 0].legend(fontsize=8, loc="lower left")

    # 0 correct rejection, 1 false positive, 2 missed edge, 3 true positive
    longest = rows[-1]
    outcome = 2 * truth.astype(int) + longest["significant"].astype(int)
    categories = ListedColormap(["#e9ecef", "#c2453f", "#8a6d1f", "#2f6f9f"])
    plot_matrix(
        axes[1, 1],
        outcome.astype(float),
        title=(
            f"Ground truth vs findings at {longest['seconds']}s\n"
            f"precision {longest['precision']:.2f}, recall {longest['recall']:.2f}"
        ),
        cmap=categories,
        vmin=-0.5,
        vmax=3.5,
        colorbar=False,
        mask_diagonal=True,
    )
    axes[1, 1].legend(
        handles=[
            Patch(color="#2f6f9f", label="true edge, found"),
            Patch(color="#8a6d1f", label="true edge, missed"),
            Patch(color="#c2453f", label="no edge, but flagged"),
            Patch(color="#e9ecef", label="no edge, not flagged"),
        ],
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    fig.suptitle("Directed coupling between realistic neural mass columns")
    save(fig, FIGURES / "neural_mass_network.png")

    print("\nresults")
    print(
        f"  {'seconds':>8} {'found':>7} {'false+':>7} {'recall':>8} {'precision':>10}"
    )
    for row in rows:
        print(
            f"  {row['seconds']:>8} {row['true_positives']:>7} "
            f"{row['false_positives']:>7} {row['recall']:>8.2f} "
            f"{row['precision']:>10.2f}"
        )

    best_recall = max(row["recall"] for row in rows)
    shortest = rows[0]

    assert 8.0 <= peak <= 13.0, (
        f"columns should oscillate in the alpha band, got {peak}"
    )
    assert alpha_fraction > 0.5, "most of the power should sit in the alpha band"
    assert shortest["recall"] < best_recall, (
        "a short recording should miss edges that a longer one finds"
    )
    assert best_recall >= 0.8, "a long enough recording should find nearly every edge"

    print(
        f"\nverified: columns oscillate at {peak:.1f} Hz with "
        f"{alpha_fraction:.0%} of power in the alpha band,"
        f"\n          and {shortest['seconds']}s of data is not enough to find "
        f"coupling that {durations[-1]}s does find."
        "\n\n          Watch precision rather than recall: the extra detections at"
        "\n          long recordings are indirect paths, not errors. For network"
        "\n          reconstruction there is an optimum recording length."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer surrogates")
    main(**vars(parser.parse_args()))
