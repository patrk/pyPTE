"""Can pyPTE reconstruct a known directed network of oscillators?

Two ground-truth topologies are simulated with delayed Kuramoto oscillators:
a directed ring, where every node drives its successor, and two internally
dense groups joined by a single one-way bridge.

The comparison is deliberately unfair to pyPTE in one respect. Ground truth is
sparse, while dPTE assigns a value to every one of the m*m ordered pairs, so the
recovered matrix is dense by construction. Rather than hide that behind a
hand-picked threshold, the score reported here is AUC: the probability that a
true edge outranks a non-edge, which needs no threshold at all.

Run with:
    uv run python -m examples.kuramoto_network [--quick]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from examples.models import kuramoto
from examples.utils.plotting import plot_matrix, save, use_house_style
from pyPTE import PTE

FIGURES = Path(__file__).resolve().parent / "figures"


def auc(scores: npt.NDArray, truth: npt.NDArray) -> float:
    """Area under the ROC curve, computed by rank comparison.

    Equivalent to the Mann-Whitney U statistic: the chance that a randomly
    chosen true edge scores above a randomly chosen non-edge. 0.5 is chance,
    1.0 is perfect separation.
    """
    positive = scores[truth]
    negative = scores[~truth]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks over ties so exact duplicates do not bias the statistic
    values = np.concatenate([positive, negative])
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    rank_sum = ranks[: positive.size].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def off_diagonal(matrix: npt.NDArray) -> npt.NDArray:
    """Flatten a matrix, dropping the diagonal, which carries no information."""
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def evaluate(coupling: npt.NDArray, *, n_samples: int, seed: int) -> dict:
    signal = kuramoto.simulate(coupling, n_samples=n_samples, seed=seed)
    dPTE, _ = PTE(signal)

    directed = kuramoto.net_flow(coupling)
    # symmetric pairs are coupled but carry no net direction, so they belong to
    # neither class: scoring them as edges would punish the correct answer
    symmetric = (coupling > 0) & (coupling == coupling.T)

    considered = off_diagonal(~symmetric)
    truth = off_diagonal(directed)[considered]
    scores = off_diagonal(dPTE)[considered]

    return {
        "dPTE": dPTE,
        "truth": directed,
        "symmetric": symmetric,
        "auc": auc(scores, truth),
        "edge_mean": scores[truth].mean(),
        "nonedge_mean": scores[~truth].mean(),
        "symmetric_mean": off_diagonal(dPTE)[off_diagonal(symmetric)].mean()
        if symmetric.any()
        else float("nan"),
    }


def main(quick: bool = False) -> None:
    use_house_style()
    n_samples = 4000 if quick else 12000

    topologies = {
        "Directed ring (i -> i+1)": kuramoto.ring_with_shortcut(8, strength=3.0),
        "Two groups, one-way bridges": kuramoto.two_groups(4, within=2.0, across=3.0),
    }

    results = {}
    for name, coupling in topologies.items():
        print(f"simulating: {name} ...")
        results[name] = evaluate(coupling, n_samples=n_samples, seed=1)

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7.2), constrained_layout=True, width_ratios=[1, 1, 1.15]
    )

    for row, (name, res) in enumerate(results.items()):
        plot_matrix(
            axes[row, 0],
            res["truth"].astype(float),
            title=f"Ground truth\n{name}",
            cmap="Greys",
            vmin=0,
            vmax=1,
            colorbar=False,
        )
        plot_matrix(
            axes[row, 1],
            res["dPTE"],
            title=f"Recovered dPTE\nAUC = {res['auc']:.3f}",
            center_on_half=True,
            mask_diagonal=True,
        )

        ax = axes[row, 2]
        flat_dPTE = off_diagonal(res["dPTE"])
        flat_directed = off_diagonal(res["truth"])
        flat_symmetric = off_diagonal(res["symmetric"])
        bins = np.linspace(0.2, 0.8, 25)

        ax.hist(
            flat_dPTE[~flat_directed & ~flat_symmetric],
            bins=bins,
            alpha=0.65,
            label="no coupling",
            color="#7f8c9b",
        )
        if flat_symmetric.any():
            ax.hist(
                flat_dPTE[flat_symmetric],
                bins=bins,
                alpha=0.7,
                label="symmetric (no net direction)",
                color="#8a6d1f",
            )
        ax.hist(
            flat_dPTE[flat_directed],
            bins=bins,
            alpha=0.8,
            label="directed edge",
            color="#c2453f",
        )
        ax.axvline(0.5, ls="--", lw=1, color="0.4")
        ax.set(xlabel="dPTE", ylabel="count", title="Score separation")
        ax.legend(fontsize=7)

    fig.suptitle("Reconstructing known directed networks of oscillators")
    save(fig, FIGURES / "kuramoto_network.png")

    print("\nresults")
    for name, res in results.items():
        print(f"  {name}")
        print(f"    AUC, directed vs uncoupled   {res['auc']:.3f}   (0.5 = chance)")
        print(f"    mean dPTE on directed edges  {res['edge_mean']:.3f}")
        print(f"    mean dPTE on uncoupled pairs {res['nonedge_mean']:.3f}")
        if not np.isnan(res["symmetric_mean"]):
            print(
                f"    mean dPTE on symmetric pairs {res['symmetric_mean']:.3f}"
                "   (expected ~0.5: coupled, but no net direction)"
            )

    for name, res in results.items():
        assert res["auc"] > 0.75, f"{name}: directed edges should outrank non-edges"
        assert res["edge_mean"] > res["nonedge_mean"], (
            f"{name}: directed edges must score above uncoupled pairs"
        )
        if not np.isnan(res["symmetric_mean"]):
            assert abs(res["symmetric_mean"] - 0.5) < 0.05, (
                f"{name}: symmetrically coupled pairs carry no net direction and "
                "should sit at 0.5"
            )

    print(
        "\nverified: directed edges outrank uncoupled pairs, while symmetrically"
        "\n          coupled pairs correctly sit at 0.5. dPTE measures direction,"
        "\n          not the presence of coupling, so a strong bidirectional link"
        "\n          is invisible to it by design."
        "\n\n          The recovered matrix is dense while the truth is sparse, so"
        "\n          turning these scores into a graph needs a significance test,"
        "\n          not a threshold picked by eye."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer samples")
    main(**vars(parser.parse_args()))
