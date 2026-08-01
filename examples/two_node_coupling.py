"""Does pyPTE recover the direction of a known one-way coupling?

The simplest possible ground truth: one oscillator drives a second with a fixed
lag, and nothing flows back. Sweeping the coupling strength and the lag shows
what dPTE does as the true influence grows, and confirms it sits at the
no-preference value of 0.5 when the two signals are independent.

The last panel is the counterweight. Two *independent* oscillators that differ
only in signal-to-noise ratio produce a large spurious dPTE, because the noisier
channel is less predictable from its own past. Read directionality from raw dPTE
alone and unequal SNR will look exactly like coupling.

Run with:
    uv run python -m examples.two_node_coupling [--quick]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.utils.plotting import save, use_house_style
from pyPTE import PTE

FS = 250.0
NOISE = 0.3
FIGURES = Path(__file__).resolve().parent / "figures"


def oscillator(rng: np.random.Generator, n: int, freq: float = 10.0) -> np.ndarray:
    """A narrowband rhythm whose phase drifts, so two of them never stay aligned.

    A pure sinusoid would be a poor stand-in for an independent rhythm: two of
    them at the same frequency hold a fixed phase offset forever, which any
    phase-based measure correctly reports as a constant lead. Diffusing the
    phase makes independent oscillators genuinely independent.
    """
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift)


def simulate(coupling: float, lag: int, n_samples: int, seed: int) -> np.ndarray:
    """Channel 0 drives channel 1 by `lag` samples with strength `coupling`.

    Both channels carry the same amount of additive noise. That symmetry is not
    cosmetic: unequal noise on its own creates apparent direction, as the final
    panel of this example demonstrates.
    """
    rng = np.random.default_rng(seed)
    driver = oscillator(rng, n_samples) + NOISE * rng.standard_normal(n_samples)

    # the target keeps a rhythm of its own, which the coupling progressively
    # displaces, so that coupling = 0 leaves two unrelated oscillators
    intrinsic = oscillator(rng, n_samples)
    target = (
        coupling * np.roll(driver, lag)
        + max(1.0 - coupling, 0.0) * intrinsic
        + NOISE * rng.standard_normal(n_samples)
    )
    return np.vstack([driver, target])


def sweep(values, *, vary: str, n_samples: int, seeds: int) -> np.ndarray:
    """Return dPTE[driver -> target] for each value, one row per seed."""
    out = np.empty((seeds, len(values)))
    for s in range(seeds):
        for k, value in enumerate(values):
            coupling = float(value) if vary == "coupling" else 0.9
            lag = 12 if vary == "coupling" else int(value)
            dPTE, _ = PTE(simulate(coupling, lag, n_samples, seed=s))
            out[s, k] = dPTE[0, 1]
    return out


def snr_bias(ratios, *, n_samples: int, seeds: int) -> np.ndarray:
    """dPTE between two INDEPENDENT oscillators whose noise levels differ."""
    out = np.empty((seeds, len(ratios)))
    for s in range(seeds):
        for k, ratio in enumerate(ratios):
            rng = np.random.default_rng(1000 + s)
            a = oscillator(rng, n_samples) + NOISE * rng.standard_normal(n_samples)
            b = oscillator(rng, n_samples) + NOISE * ratio * rng.standard_normal(
                n_samples
            )
            dPTE, _ = PTE(np.vstack([a, b]))
            out[s, k] = dPTE[0, 1]
    return out


def _trend(ax, x, data, *, color, xlabel, title, ylim=(0.35, 1.0)):
    mean, sd = data.mean(0), data.std(0)
    ax.plot(x, mean, "o-", color=color, lw=1.5, ms=4)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18)
    ax.axhline(0.5, ls="--", lw=1, color="0.4")
    ax.set(xlabel=xlabel, ylabel="dPTE (A -> B)", title=title, ylim=ylim)


def main(quick: bool = False) -> None:
    use_house_style()
    n_samples = 3000 if quick else 8000
    seeds = 2 if quick else 5

    couplings = np.linspace(0.0, 1.4, 6 if quick else 10)
    lags = np.arange(2, 30, 6 if quick else 3)
    ratios = np.linspace(1.0, 4.0, 4 if quick else 7)

    print("sweeping coupling strength ...")
    by_coupling = sweep(couplings, vary="coupling", n_samples=n_samples, seeds=seeds)
    print("sweeping transmission lag ...")
    by_lag = sweep(lags, vary="lag", n_samples=n_samples, seeds=seeds)
    print("sweeping SNR asymmetry between independent signals ...")
    by_ratio = snr_bias(ratios, n_samples=n_samples, seeds=seeds)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), constrained_layout=True)

    signal = simulate(0.9, 12, n_samples, seed=0)
    window = np.arange(200)
    axes[0, 0].plot(window / FS, signal[0, :200], lw=1.1, label="A (driver)")
    axes[0, 0].plot(window / FS, signal[1, :200], lw=1.1, label="B (target)")
    axes[0, 0].set(xlabel="time (s)", ylabel="amplitude", title="Signals: A leads B")
    axes[0, 0].legend(loc="upper right", ncols=2, fontsize=8)

    _trend(
        axes[0, 1],
        couplings,
        by_coupling,
        color="#c2453f",
        xlabel="coupling strength",
        title="Recovered direction vs coupling",
    )
    _trend(
        axes[1, 0],
        lags,
        by_lag,
        color="#2f6f9f",
        xlabel="lag (samples)",
        title="Recovered direction vs lag",
    )
    _trend(
        axes[1, 1],
        ratios,
        by_ratio,
        color="#8a6d1f",
        xlabel="noise ratio  B / A",
        title="CAVEAT: independent signals, unequal SNR",
    )
    axes[1, 1].text(
        ratios[0],
        0.93,
        "no coupling exists here",
        fontsize=8,
        color="#8a6d1f",
        style="italic",
    )

    fig.suptitle("pyPTE recovers known coupling, and what can fake it")
    save(fig, FIGURES / "two_node_coupling.png")

    uncoupled = by_coupling[:, 0].mean()
    strongest = by_coupling[:, -1].mean()
    reverse, _ = PTE(simulate(0.9, 12, n_samples, seed=0))
    worst_bias = by_ratio[:, -1].mean()

    print("\nresults")
    print(f"  uncoupled, equal noise   dPTE = {uncoupled:.3f}   (expect ~0.5)")
    print(f"  strongest coupling       dPTE = {strongest:.3f}   (expect > 0.5)")
    print(f"  reverse direction        dPTE = {reverse[1, 0]:.3f}   (expect < 0.5)")
    print(
        f"  independent, {ratios[-1]:.0f}x noise  dPTE = {worst_bias:.3f}   (SPURIOUS)"
    )

    assert abs(uncoupled - 0.5) < 0.05, "uncoupled signals should show no direction"
    assert strongest > 0.6, "strong coupling should be clearly directional"
    assert reverse[1, 0] < 0.5, "the reverse direction must be suppressed"
    assert (by_lag > 0.5).all(), "direction should be recovered at every lag tested"
    assert worst_bias > 0.6, (
        "the SNR caveat should be visible; if this fails the demonstration of the "
        "limitation is no longer valid"
    )

    print(
        "\nverified: direction is recovered when it exists and absent when it does not,"
        "\n          but unequal SNR alone produces a comparable dPTE, so raw values"
        "\n          must be tested against surrogates before they mean anything."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer samples and seeds")
    main(**vars(parser.parse_args()))
