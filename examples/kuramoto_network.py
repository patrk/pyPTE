"""
Can pyPTE reconstruct a known directed network of oscillators?
==============================================================

Two ground-truth topologies are simulated with delayed Kuramoto oscillators:
a directed ring, where every node drives its successor, and two internally
dense groups joined by one-way bridges. Each is run over 20 noise
realisations, because one realisation can flatter or sink the score.

Ground truth is sparse but dPTE gives every ordered pair a value, so the
recovered matrix is dense no matter what. Scored with AUC rather than a
hand-picked threshold: the chance a true edge outranks a non-edge.

The ring is recovered reliably (AUC 0.95 +- 0.03). The two groups are not
(0.84 +- 0.10, as low as 0.59): a bridge's input spreads through its whole,
tightly synchronised group, so every A -> B pair carries real flow. Bridges
outrank their own reverse almost always (0.97) but outrank those indirect
paths only 0.67 of the time.

Symmetric coupling does not read as 0.5 either. The oscillators' natural
frequencies differ, and that alone gives each symmetric pair a direction,
on average 0.11 from 0.5 against 0.19 for a bridge; with identical
frequencies it drops to 0.02.

Run with::

    uv run python -m examples.kuramoto_network [--quick]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from examples.models import kuramoto
from examples.utils import style
from examples.utils.plotting import FIGURES, plot_matrix, save, use_house_style
from pyPTE import PTE


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
    # symmetric pairs have no direction in the ground truth, so they belong to
    # neither class
    symmetric = (coupling > 0) & (coupling == coupling.T)
    # pairs a directed path joins without an edge of their own: real flow, but
    # indirect, so a non-edge that should not be expected to score at 0.5
    reach = np.linalg.matrix_power(
        np.eye(len(coupling)) + (coupling > 0), len(coupling)
    )
    indirect = (reach > 0) & ~directed & ~directed.T & ~symmetric
    np.fill_diagonal(indirect, False)

    considered = off_diagonal(~symmetric)
    truth = off_diagonal(directed)[considered]
    scores = off_diagonal(dPTE)[considered]
    # dPTE[i, j] + dPTE[j, i] = 1, so a mean over both orientations of a pair
    # is 0.5 whatever happens; the distance from 0.5 is what can be judged
    distance = np.abs(dPTE - 0.5)

    return {
        "dPTE": dPTE,
        "truth": directed,
        "symmetric": symmetric,
        "indirect": indirect,
        "auc": auc(scores, truth),
        "auc_indirect": auc(
            np.concatenate([dPTE[directed], dPTE[indirect]]),
            np.arange(directed.sum() + indirect.sum()) < directed.sum(),
        ),
        "auc_reverse": auc(
            np.concatenate([dPTE[directed], dPTE[directed.T]]),
            np.arange(2 * directed.sum()) < directed.sum(),
        ),
        "edge_distance": distance[directed].mean(),
        "symmetric_distance": distance[symmetric].mean()
        if symmetric.any()
        else float("nan"),
    }


def summarise(runs: list[dict], key: str) -> npt.NDArray:
    return np.array([run[key] for run in runs])


def main(quick: bool = False) -> None:
    use_house_style()
    n_samples = 4000 if quick else 12000
    seeds = range(1, 6) if quick else range(1, 21)

    topologies = {
        "Directed ring (i -> i+1)": kuramoto.ring_with_shortcut(8, strength=3.0),
        "Two groups, one-way bridges": kuramoto.two_groups(4, within=2.0, across=3.0),
    }

    runs = {}
    for name, coupling in topologies.items():
        print(f"simulating: {name}, {len(seeds)} realisations ...")
        runs[name] = [evaluate(coupling, n_samples=n_samples, seed=s) for s in seeds]

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7.2), constrained_layout=True, width_ratios=[1, 1, 1.15]
    )

    for row, (name, realisations) in enumerate(runs.items()):
        aucs = summarise(realisations, "auc")
        # show the median realisation, not the best one
        res = realisations[int(np.argsort(aucs)[len(aucs) // 2])]
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
            title=f"Median realisation, AUC = {res['auc']:.2f}\n"
            f"over {len(aucs)}: {aucs.mean():.2f} ± {aucs.std():.2f}",
            center_on_half=True,
            mask_diagonal=True,
        )

        ax = axes[row, 2]
        flat_dPTE = off_diagonal(res["dPTE"])
        flat_directed = off_diagonal(res["truth"])
        flat_symmetric = off_diagonal(res["symmetric"])
        flat_indirect = off_diagonal(res["indirect"])
        bins = np.linspace(0.0, 1.0, 41)

        for members, label, color in [
            (
                ~(flat_directed | flat_symmetric | flat_indirect),
                "other pairs",
                style.NULL,
            ),
            (flat_symmetric, "symmetric coupling", style.AMBER),
            (flat_indirect, "indirect path", style.BLUE),
            (flat_directed, "directed edge", style.OBS),
        ]:
            if members.any():
                ax.hist(
                    flat_dPTE[members], bins=bins, alpha=0.7, label=label, color=color
                )
        ax.axvline(0.5, ls="--", lw=1, color="0.4")
        ax.set(xlabel="dPTE", ylabel="count", title="Score separation")
        ax.legend(fontsize=7, loc="upper center" if row else "upper right")

    save(fig, FIGURES / "kuramoto_network.png")

    ring, groups = runs.values()
    print(f"\nresults over {len(seeds)} realisations, mean ± sd (min)")
    for name, realisations in runs.items():
        print(f"  {name}")
        for key, label in [
            ("auc", "AUC, edges vs all other pairs"),
            ("auc_reverse", "AUC, edges vs their reverse"),
            ("auc_indirect", "AUC, edges vs indirect paths"),
            ("edge_distance", "|dPTE - 0.5| on edges"),
            ("symmetric_distance", "|dPTE - 0.5| on symmetric pairs"),
        ]:
            values = summarise(realisations, key)
            if not np.isnan(values).all():
                print(
                    f"    {label:<32} {values.mean():.3f} ± {values.std():.3f}"
                    f"  ({values.min():.3f})"
                )
        aucs = summarise(realisations, "auc")
        print(f"    realisations with AUC below 0.75: {(aucs < 0.75).sum()}")

    ring_auc = summarise(ring, "auc")
    group_auc = summarise(groups, "auc")
    assert ring_auc.mean() > 0.9 and ring_auc.min() > 0.8, (
        "every realisation of the ring should rank its edges well"
    )
    assert summarise(groups, "auc_reverse").mean() > 0.9, (
        "bridges should be reported the right way round"
    )
    assert summarise(groups, "auc_indirect").mean() < 0.85, (
        "indirect A -> B paths should compete with the bridges"
    )
    symmetric = summarise(groups, "symmetric_distance").mean()
    assert 0.05 < symmetric < summarise(groups, "edge_distance").mean(), (
        "detuned symmetric pairs should read as directed, if less than bridges"
    )

    print(
        "\nverified: the ring's edges are recovered in every realisation, and"
        "\n          bridges are reported the right way round. But an edge into a"
        "\n          synchronised group looks like an edge into each member, and"
        "\n          symmetric coupling between detuned oscillators reads as"
        "\n          directed. dPTE ranks directed paths, not edges."
        "\n\n          One realisation is not a result: single realisations of the"
        f"\n          two groups scored anywhere from {group_auc.min():.2f} to"
        f" {group_auc.max():.2f}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer samples")
    main(**vars(parser.parse_args()))
