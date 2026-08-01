"""Run pyPTE against a real anatomical connectome, and mostly fail.

Uses The Virtual Brain's 76-region tract-tracing connectome, so the
directionality is real anatomy rather than something we invented. pyPTE only
reaches AUC 0.62 here, against 0.96 to 1.00 on the sparse synthetic networks in
the other examples. The connectome is 27% dense, so nearly every pair of regions
is joined by a short indirect path, and dPTE ranks indirect paths above direct
ones.

Scale also makes edgewise significance testing impractical: 5700 pairs need
p below 1e-5, but N surrogates cannot go below 1/(N+1).

Needs the optional TVB group:
    uv sync --group examples --group tvb
    uv run python -m examples.tvb_connectome [--quick]
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.stats import rankdata

from examples.utils.plotting import plot_matrix, save, use_house_style
from pyPTE import PTE, surrogate_test

FS = 250.0  # the temporal-average monitor below samples every 4 ms
COUPLING = 0.1
FIGURES = Path(__file__).resolve().parent / "figures"


def load_tvb():
    """Import TVB, with a clear message if the optional group is missing."""
    try:
        # TVB is chatty about a missing optional field in its own data file
        warnings.filterwarnings("ignore")
        from tvb.simulator.lab import (
            connectivity,
            coupling,
            integrators,
            models,
            monitors,
            noise,
            simulator,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "This example needs The Virtual Brain:\n"
            "    uv sync --group examples --group tvb"
        ) from exc
    return simulator, models, connectivity, coupling, integrators, monitors, noise


def auc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Probability a true edge outranks a non-edge; 0.5 is chance."""
    positive, negative = scores[truth], scores[~truth]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    return float(
        (ranks[: positive.size].sum() - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def simulate(seconds: float):
    """Run Generic2dOscillator on the bundled connectome.

    Jansen-Rit diverges to NaN on this connectome at TVB's default coupling, so
    the generic oscillator is used instead; it produces a stable rhythm without
    per-region parameter tuning.
    """
    simulator, models, connectivity, coupling, integrators, monitors, noise = load_tvb()

    brain = connectivity.Connectivity.from_file()
    brain.configure()
    brain.speed = np.array([10.0])

    sim = simulator.Simulator(
        model=models.Generic2dOscillator(),
        connectivity=brain,
        coupling=coupling.Linear(a=np.array([COUPLING])),
        integrator=integrators.HeunStochastic(
            dt=0.5, noise=noise.Additive(nsig=np.array([1e-4]))
        ),
        monitors=(monitors.TemporalAverage(period=1000.0 / FS),),
    )
    sim.configure()
    ((_, data),) = sim.run(simulation_length=seconds * 1000.0)

    # (time, state_var, region, mode) -> (region, time)
    signal = np.asarray(data)[:, 0, :, 0].T
    return brain, signal


def main(quick: bool = False) -> None:
    use_house_style()
    seconds = 40.0 if quick else 120.0

    print(f"simulating {seconds:.0f}s of whole-brain activity ...")
    brain, signal = simulate(seconds)
    weights = brain.weights
    n_regions = weights.shape[0]

    freqs, power = welch(signal[0], fs=FS, nperseg=512)
    peak = float(freqs[np.argmax(power)])
    print(f"  {n_regions} regions, {signal.shape[1]} samples, peak {peak:.2f} Hz")

    off_diagonal = ~np.eye(n_regions, dtype=bool)
    one_way = (weights > 0) & (weights.T == 0)
    unconnected = off_diagonal & (weights == 0) & (weights.T == 0)
    symmetric = (weights > 0) & (weights == weights.T)
    density = float((weights > 0).sum() / off_diagonal.sum())

    print("running PTE ...")
    dPTE, _ = PTE(signal)

    considered = one_way | unconnected
    score = auc(dPTE[considered], one_way[considered])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    plot_matrix(
        axes[0, 0],
        (weights > 0).astype(float),
        title=f"TVB connectome: which regions connect\ndensity {density:.0%}",
        cmap="Greys",
        vmin=0,
        vmax=1,
        colorbar=False,
    )
    plot_matrix(
        axes[0, 1],
        one_way.astype(float),
        title=f"Edges existing in one direction only\n{int(one_way.sum())} of "
        f"{int((weights > 0).sum())}",
        cmap="Greys",
        vmin=0,
        vmax=1,
        colorbar=False,
    )
    plot_matrix(
        axes[1, 0],
        dPTE,
        title=f"Recovered dPTE\nAUC vs one-way edges = {score:.3f}",
        center_on_half=True,
        mask_diagonal=True,
    )

    ax = axes[1, 1]
    bins = np.linspace(0.3, 0.7, 40)
    ax.hist(
        dPTE[unconnected],
        bins=bins,
        alpha=0.6,
        density=True,
        label="unconnected",
        color="#7f8c9b",
    )
    ax.hist(
        dPTE[one_way],
        bins=bins,
        alpha=0.75,
        density=True,
        label="one-way edge",
        color="#c2453f",
    )
    ax.axvline(0.5, ls="--", lw=1, color="0.4")
    ax.set(
        xlabel="dPTE",
        ylabel="density",
        title="The two distributions barely separate",
    )
    ax.legend(fontsize=8)

    fig.suptitle("pyPTE against a real anatomical connectome (The Virtual Brain)")
    save(fig, FIGURES / "tvb_connectome.png")

    # a second, practical obstacle at this scale: the surrogate count sets a
    # floor of 1 / (n_surrogates + 1) on any p-value, while FDR over m*(m-1)
    # pairs demands far smaller ones
    n_pairs = int(off_diagonal.sum())
    n_surrogates = 60
    print(f"surrogate testing with {n_surrogates} surrogates ...")
    significance = surrogate_test(signal, n_surrogates=n_surrogates, seed=0)
    floor = 1.0 / (n_surrogates + 1)
    strictest = 0.05 / n_pairs

    print("\nresults")
    print(f"  regions                       {n_regions}")
    print(f"  network density               {density:.1%}")
    print(f"  one-way edges                 {int(one_way.sum())}")
    print(f"  symmetric pairs               {int(symmetric.sum())}")
    print(f"  AUC, one-way vs unconnected   {score:.3f}   (0.5 = chance)")
    print(f"  mean dPTE on one-way edges    {dPTE[one_way].mean():.3f}")
    print(f"  mean dPTE on unconnected      {dPTE[unconnected].mean():.3f}")
    print(f"  edges surviving surrogates    {significance.n_significant} of {n_pairs}")
    print(f"  smallest p reachable          {floor:.4f}  (1 / (surrogates + 1))")
    print(f"  strictest FDR threshold       {strictest:.2e}  (0.05 / {n_pairs})")

    # data integrity: the point of using TVB is that this connectome is directed
    assert not np.allclose(weights, weights.T), (
        "the TVB connectome should be asymmetric; if this fails the example has "
        "lost the only property that makes it worth running"
    )
    assert int(one_way.sum()) > 100, "expected a few hundred one-way edges"
    assert np.isfinite(signal).all(), "the simulation diverged"
    assert 1.0 < peak < 40.0, f"expected an oscillatory signal, got {peak:.1f} Hz"

    # the finding itself: recovery is poor. This guard exists so that if a future
    # change makes it good, the documentation above gets revisited rather than
    # silently becoming wrong.
    assert score < 0.75, (
        f"AUC rose to {score:.3f}; structural recovery is documented as close to "
        "chance, so that text needs updating"
    )

    print(
        f"\nverified: the connectome is genuinely directed ({int(one_way.sum())} "
        "one-way edges),"
        f"\n          the simulation is stable and oscillatory at {peak:.1f} Hz,"
        f"\n          and pyPTE separates real edges from non-edges at AUC "
        f"{score:.3f}."
        "\n\n          That is weak, and it is the expected outcome: at "
        f"{density:.0%} density"
        "\n          nearly every region pair is joined by a short indirect path,"
        "\n          and dPTE ranks indirect paths above direct ones. Compare the"
        "\n          AUC of 0.96 to 1.00 the same estimator reaches on sparse"
        "\n          synthetic networks in kuramoto_network.py."
        f"\n\n          Scale bites twice: {n_pairs} ordered pairs need p below "
        f"{strictest:.1e},"
        f"\n          but {n_surrogates} surrogates cannot produce a p below "
        f"{floor:.3f}, so"
        f"\n          {significance.n_significant} edges survive. Whole-brain "
        "surrogate testing needs"
        "\n          thousands of surrogates, or a cluster-level test instead."
        "\n\n          Use pyPTE to ask which way activity flows, not to infer anatomy."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="shorter simulation")
    main(**vars(parser.parse_args()))
