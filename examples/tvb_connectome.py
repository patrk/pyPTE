"""
Run pyPTE against a real anatomical connectome, and mostly fail.
================================================================

Uses The Virtual Brain's 76-region tract-tracing connectome, so the
directionality is real anatomy rather than something we invented. Band-passed
and analysed with time reversal, dPTE sees that two regions are connected but
not which way: how far it sits from 0.5, on either side, separates one-way
tracts from unconnected pairs at an AUC of about 0.78, while its sign does so
at about 0.53, barely above chance. The connectome is 27% dense, so nearly
every pair of regions is also joined by short indirect paths, which can run
either way.

Plain dPTE does worse than chance, at about 0.39. Regions that receive more
input oscillate more strongly, which makes their rhythm cleaner, and dPTE reads
the cleaner channel as the driver: the opposite of the anatomy. Time reversal
cancels that.

Scale also makes edgewise significance testing impractical: 2850 channel pairs
need p below 2e-5, but N surrogates cannot go below 1/(N+1).

Needs the optional TVB group:
    uv sync --group examples --group tvb
    uv run python -m examples.tvb_connectome [--quick]
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import rankdata, spearmanr

from examples.utils import style
from examples.utils.plotting import FIGURES, plot_matrix, save, use_house_style
from pyPTE import PTE, surrogate_test

FS = 250.0  # the temporal-average monitor below samples every 4 ms
COUPLING = 0.1


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


def tracts(brain) -> np.ndarray:
    """Tract strengths with ``[i, j]`` the connection from region i to j.

    TVB stores ``weights[to, from]``: its coupling sums ``weights[i, j]``
    times region j's activity into region i. dPTE's ``[i, j]`` is flow from i
    to j, so the anatomy is the transpose.
    """
    return np.asarray(brain.weights).T


def narrow_band(signal: np.ndarray) -> np.ndarray:
    """Band-pass every region to 1.5 Hz either side of the shared peak."""
    freqs, power = welch(signal, fs=FS, nperseg=1024)
    peak = float(freqs[np.argmax(power.mean(axis=0))])
    band = butter(4, [peak - 1.5, peak + 1.5], btype="band", fs=FS, output="sos")
    return sosfiltfilt(band, signal, axis=1, padlen=int(FS))


def flow(signal: np.ndarray) -> np.ndarray:
    """dPTE as it should be run: band-passed, time-reversed, bins for direction."""
    dPTE, _ = PTE(narrow_band(signal), time_reversal=True, binning="locking")
    return dPTE


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
    weights = tracts(brain)
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
    dPTE = flow(signal)
    plain, _ = PTE(signal)

    considered = one_way | unconnected
    score = auc(dPTE[considered], one_way[considered])
    plain_score = auc(plain[considered], one_way[considered])
    # how far from 0.5, ignoring which side: does dPTE see a tract at all?
    strength = np.abs(dPTE - 0.5)
    existence_score = auc(strength[considered], one_way[considered])
    # input received, against how strongly each region oscillates
    amplitude_rho = float(spearmanr(weights.sum(axis=0), signal.std(axis=1)).statistic)

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
        title=(
            "Recovered dPTE, band-passed, time-reversed\n"
            f"AUC vs one-way edges = {score:.3f}"
        ),
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
        color=style.NULL,
    )
    ax.hist(
        dPTE[one_way],
        bins=bins,
        alpha=0.75,
        density=True,
        label="one-way edge",
        color=style.OBS,
    )
    ax.axvline(0.5, ls="--", lw=1, color="0.4")
    ax.set(
        xlabel="dPTE",
        ylabel="density",
        title=(
            f"One-way tracts sit away from 0.5, on either side\n"
            f"AUC {existence_score:.2f} ignoring the side, {score:.2f} with it"
        ),
    )
    ax.legend(fontsize=8)

    save(fig, FIGURES / "tvb_connectome.png")

    # a second, practical obstacle at this scale: the surrogate count sets a
    # floor of 1 / (n_surrogates + 1) on any p-value, while FDR over this many
    # pairs demands far smaller ones. dPTE in one direction fixes the other, so
    # Benjamini-Hochberg in effect runs over channel pairs, half the ordered ones
    n_pairs = int(off_diagonal.sum()) // 2
    n_surrogates = 60
    print(f"surrogate testing with {n_surrogates} surrogates ...")
    significance = surrogate_test(
        narrow_band(signal),
        n_surrogates=n_surrogates,
        seed=0,
        binning="locking",
        time_reversal=True,
    )
    floor = 1.0 / (n_surrogates + 1)
    strictest = 0.05 / n_pairs
    survived = int(significance.directed.sum())

    print("\nresults")
    print(f"  regions                       {n_regions}")
    print(f"  network density               {density:.1%}")
    print(f"  one-way edges                 {int(one_way.sum())}")
    print(f"  symmetric pairs               {int(symmetric.sum())}")
    print(f"  AUC, one-way vs unconnected   {score:.3f}   (0.5 = chance)")
    print(f"  AUC with plain dPTE           {plain_score:.3f}   (below chance)")
    print(f"  AUC of |dPTE - 0.5|           {existence_score:.3f}   (either way)")
    print(f"  input strength vs amplitude   rho {amplitude_rho:.2f}")
    print(f"  mean dPTE on one-way edges    {dPTE[one_way].mean():.3f}")
    print(f"  mean dPTE on unconnected      {dPTE[unconnected].mean():.3f}")
    print(f"  pairs surviving surrogates    {survived} of {n_pairs}")
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
    assert existence_score > 0.65, "dPTE should still see that regions connect"
    # and plain dPTE points against the anatomy, through amplitude
    assert plain_score < 0.5, "plain dPTE should rank tracts below chance"
    assert amplitude_rho > 0.8, "amplitude should follow the input a region receives"

    print(
        f"\nverified: the connectome is genuinely directed ({int(one_way.sum())} "
        "one-way edges),"
        f"\n          the simulation is stable and oscillatory at {peak:.1f} Hz,"
        "\n          and dPTE sees which regions connect (AUC "
        f"{existence_score:.3f} on |dPTE - 0.5|)"
        f"\n          but not which way: its sign scores {score:.3f}."
        f"\n\n          At {density:.0%} density nearly every region pair is joined"
        "\n          by short indirect paths, which can run either way."
        f"\n          Plain dPTE scores {plain_score:.3f}, below chance: regions that"
        "\n          receive more input oscillate more cleanly and read as drivers."
        f"\n\n          Scale bites twice: {n_pairs} channel pairs need p below "
        f"{strictest:.1e},"
        f"\n          but {n_surrogates} surrogates cannot produce a p below "
        f"{floor:.3f}, so"
        f"\n          {survived} pairs survive. Whole-brain surrogate testing needs"
        "\n          thousands of surrogates, or a cluster-level test instead."
        "\n\n          Use pyPTE to ask which way activity flows, not to infer anatomy."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="shorter simulation")
    main(**vars(parser.parse_args()))
