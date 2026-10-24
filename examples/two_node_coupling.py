"""
Does pyPTE recover the direction of a known one-way coupling?
=============================================================

The simplest ground truth there is: one oscillator drives a second with a fixed
lag and nothing flows back. Both channels carry the same noise, are band-passed
to the rhythm, and are analysed with time reversal and bins sized for the
direction (``binning="locking"``). Sweeps coupling strength and lag, and checks
that dPTE sits at 0.5 when the two are independent.

The last panel is the catch. Two independent oscillators that differ only in
signal-to-noise ratio produce a large spurious dPTE, because the noisier channel
is harder to predict from its own past. Plain dPTE reads that gap as coupling;
the time-reversal-differenced dPTE does not.

Run with::

    uv run python -m examples.two_node_coupling [--quick]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from examples.utils import style
from examples.utils.plotting import FIGURES, save, use_house_style
from pyPTE import PTE

FS = 250.0
NOISE = 0.5
PAD = 500  # filtered and then dropped, so the filter's edges stay out
ALPHA = butter(4, [8.0, 12.0], btype="band", fs=FS, output="sos")
SETTINGS = {"time_reversal": True, "binning": "locking"}


def oscillator(rng: np.random.Generator, n: int, freq: float = 10.0) -> np.ndarray:
    """A narrowband rhythm whose phase drifts, so two of them never stay aligned.

    A pure sinusoid would be a poor stand-in for an independent rhythm: two of
    them at the same frequency hold a fixed phase offset forever, which any
    phase-based measure correctly reports as a constant lead. Diffusing the
    phase makes independent oscillators genuinely independent.
    """
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    start = rng.uniform(0.0, 2 * np.pi)
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift + start)


def band_pass(channels: np.ndarray, n_samples: int) -> np.ndarray:
    return sosfiltfilt(ALPHA, channels, axis=1)[:, PAD : PAD + n_samples]


def simulate(coupling: float, lag: int, n_samples: int, seed: int) -> np.ndarray:
    """Channel 0 drives channel 1 by `lag` samples, with a share `coupling`.

    The target's rhythm is that share of the driver's, `lag` samples late,
    and the rest a rhythm of its own, scaled to the driver's amplitude. Both
    channels then get the same noise, and only the driver's rhythm, not its
    noise, reaches the target: unequal noise on its own creates apparent
    direction, as the final panel shows.
    """
    rng = np.random.default_rng(seed)
    n = n_samples + 2 * PAD
    # the lag is cut from a longer rhythm rather than rolled, which would wrap
    # the end of the recording round to its start
    rhythm = oscillator(rng, n + lag)
    driver = rhythm[lag:]
    mixed = coupling * rhythm[:n] + (1.0 - coupling) * oscillator(rng, n)
    target = mixed * driver.std() / mixed.std()
    channels = np.vstack([driver, target]) + NOISE * rng.standard_normal((2, n))
    return band_pass(channels, n_samples)


def sweep(values, *, vary: str, n_samples: int, seeds: int) -> np.ndarray:
    """Return dPTE[driver -> target] for each value, one row per seed."""
    out = np.empty((seeds, len(values)))
    for s in range(seeds):
        for k, value in enumerate(values):
            coupling = float(value) if vary == "coupling" else 0.9
            lag = 12 if vary == "coupling" else int(value)
            dPTE, _ = PTE(simulate(coupling, lag, n_samples, seed=s), **SETTINGS)
            out[s, k] = dPTE[0, 1]
    return out


def snr_bias(ratios, *, n_samples: int, seeds: int) -> np.ndarray:
    """dPTE, plain and differenced, between two INDEPENDENT oscillators whose
    noise levels differ. Returns seeds x ratios x 2."""
    out = np.empty((seeds, len(ratios), 2))
    n = n_samples + 2 * PAD
    for s in range(seeds):
        for k, ratio in enumerate(ratios):
            rng = np.random.default_rng(1000 + s)
            a = oscillator(rng, n) + 0.3 * rng.standard_normal(n)
            b = oscillator(rng, n) + 0.3 * ratio * rng.standard_normal(n)
            pair = band_pass(np.vstack([a, b]), n_samples)
            for column, reverse in enumerate((False, True)):
                dPTE, _ = PTE(pair, binning="locking", time_reversal=reverse)
                out[s, k, column] = dPTE[0, 1]
    return out


def _trend(ax, x, data, *, color, xlabel, title, ylim=(0.35, 1.0)):
    mean, sd = data.mean(0), data.std(0)
    ax.plot(x, mean, "o-", color=color, lw=1.5, ms=4)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18)
    ax.axhline(0.5, ls="--", lw=1, color="0.4")
    ax.set(xlabel=xlabel, ylabel="dPTE (A -> B)", title=title, ylim=ylim)


def main(quick: bool = False) -> None:
    use_house_style()
    n_samples = 8000 if quick else 16000
    seeds = 2 if quick else 5

    couplings = np.linspace(0.0, 1.0, 5 if quick else 9)
    lags = np.arange(2, 30, 6 if quick else 3)
    ratios = np.linspace(1.0, 8.0, 4 if quick else 8)

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
        color=style.OBS,
        xlabel="coupling strength",
        title="Recovered direction vs coupling",
    )
    _trend(
        axes[1, 0],
        lags,
        by_lag,
        color=style.BLUE,
        xlabel="lag (samples)",
        title="Recovered direction vs lag",
    )
    _trend(
        axes[1, 1],
        ratios,
        by_ratio[:, :, 0],
        color=style.AMBER,
        xlabel="noise ratio  B / A",
        title="CAVEAT: independent signals, unequal SNR",
    )
    axes[1, 1].plot(
        ratios, by_ratio[:, :, 1].mean(0), "s-", color=style.BLUE, lw=1.5, ms=4
    )
    axes[1, 1].text(ratios[-1], 0.9, "plain dPTE", color=style.AMBER, ha="right")
    axes[1, 1].text(
        ratios[-1], 0.42, "with time reversal", color=style.BLUE, ha="right"
    )
    axes[1, 1].text(
        ratios[0],
        0.95,
        "no coupling exists here",
        fontsize=8,
        color=style.AMBER,
        style="italic",
    )

    save(fig, FIGURES / "two_node_coupling.png")

    uncoupled = by_coupling[:, 0].mean()
    strongest = by_coupling[:, -1].mean()
    reverse, _ = PTE(simulate(0.9, 12, n_samples, seed=0), **SETTINGS)
    worst_plain, worst_reversed = by_ratio[:, -1].mean(axis=0)
    lag_means = by_lag.mean(axis=0)

    print("\nresults")
    print(f"  uncoupled, equal noise   dPTE = {uncoupled:.3f}   (expect ~0.5)")
    print(f"  full coupling            dPTE = {strongest:.3f}   (expect > 0.5)")
    print(f"  reverse direction        dPTE = {reverse[1, 0]:.3f}   (expect < 0.5)")
    by_lag_text = "  ".join(
        f"{lag}:{value:.2f}" for lag, value in zip(lags, lag_means, strict=True)
    )
    print(f"  by lag  {by_lag_text}")
    print(
        f"  independent, {ratios[-1]:.0f}x noise  plain dPTE = {worst_plain:.3f} "
        f"(SPURIOUS), with time reversal {worst_reversed:.3f}"
    )

    assert abs(uncoupled - 0.5) < 0.05, "uncoupled signals should show no direction"
    assert strongest > 0.6, "strong coupling should be clearly directional"
    assert reverse[1, 0] < 0.5, "the reverse direction must be suppressed"
    # a lag of a few samples moves dPTE only slightly off 0.5; longer lags
    # carry more of the driver's drift and read as stronger direction
    assert (lag_means[lags >= 5] > 0.5).all(), "direction should show from lag 5"
    assert lag_means[-1] > lag_means[lags >= 5][0] + 0.1, "longer lags read stronger"
    assert worst_plain > 0.6, (
        "the SNR caveat should be visible; if this fails the demonstration of the "
        "limitation is no longer valid"
    )
    assert abs(worst_reversed - 0.5) < 0.05, "time reversal should cancel the gap"

    print(
        "\nverified: direction is recovered when it exists and absent when it does not."
        "\n          Unequal SNR alone produces a comparable plain dPTE; time"
        "\n          reversal cancels it, and a surrogate test rejects it."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer samples and seeds")
    main(**vars(parser.parse_args()))
