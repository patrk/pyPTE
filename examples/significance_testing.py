"""
What a significance test actually does to a dPTE matrix.
========================================================

Three channels, band-passed to the rhythm: A really drives C with a lag, and
the two carry the same noise; B is independent of everything but eight times
noisier. In the raw matrix both A -> C and A -> B look like findings.

Against its own null the artefact is usually rejected, because B's noise
raises its surrogate distribution just as much as the observed value: in 8 of
10 realisations. Not in the one shown, the first, where it survives alongside
the real coupling. Time reversal is what separates them every time: a real
direction reverses with the data, one made by a noise gap does not, so A -> C
is confirmed in all ten realisations and A -> B in none.

Run with::

    uv run python -m examples.significance_testing [--quick]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from examples.utils import style
from examples.utils.plotting import FIGURES, plot_matrix, save, use_house_style
from pyPTE import surrogate_test

FS = 250.0
LAG = 12
PAD = 500  # filtered and then dropped, so the filter's edges stay out
ALPHA = butter(4, [8.0, 12.0], btype="band", fs=FS, output="sos")
CHANNELS = ["A (driver)", "B (noisy, independent)", "C (driven by A)"]


def oscillator(rng: np.random.Generator, n: int, freq: float = 10.0) -> np.ndarray:
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    start = rng.uniform(0.0, 2 * np.pi)
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift + start)


def simulate(n_samples: int, seed: int) -> np.ndarray:
    """A -> C is real. B is independent, and noisy enough to look coupled."""
    rng = np.random.default_rng(seed)
    n = n_samples + 2 * PAD

    # C carries A's rhythm LAG samples late, cut from a longer rhythm rather
    # than rolled, with noise of its own equal to A's
    rhythm = oscillator(rng, n + LAG)
    driver = rhythm[LAG:] + 0.3 * rng.standard_normal(n)
    driven = rhythm[:n] + 0.3 * rng.standard_normal(n)
    # eight times the noise of A, and no relationship to it whatsoever
    noisy = oscillator(rng, n) + 2.4 * rng.standard_normal(n)

    channels = np.vstack([driver, noisy, driven])
    return sosfiltfilt(ALPHA, channels, axis=1)[:, PAD : PAD + n_samples]


def plot_null(ax, result, i: int, j: int, *, title: str) -> None:
    """Where the observation falls inside its own surrogate distribution."""
    null = result.null_distribution[:, i, j]
    observed = result.dPTE[i, j]
    p = result.p_values[i, j]
    survives = bool(result.significant[i, j])
    confirmed = bool(result.reversal_confirmed[i, j])

    colour = style.BLUE if survives and confirmed else style.OBS
    ax.hist(null, bins=20, color=style.NULL, label="surrogate null")
    ax.axvline(
        observed,
        color=colour,
        lw=2.2,
        label=f"observed = {observed:.3f}",
    )
    ax.axvline(0.5, ls="--", lw=1, color="0.5")
    ax.set(xlabel="dPTE", ylabel="surrogates", title=title)
    ax.legend(fontsize=7, loc="upper left")
    ax.text(
        0.02,
        0.80,
        f"p = {p:.3f}, {'significant' if survives else 'not significant'}\n"
        f"time reversal: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        fontsize=8,
        color=colour,
        weight="bold",
    )


def main(quick: bool = False) -> None:
    use_house_style()
    # The surrogate count sets a floor on the p-values available: with n
    # surrogates nothing can score below 1 / (n + 1). Three channels give six
    # ordered pairs, so the strictest Benjamini-Hochberg threshold is
    # 0.05 / 6 = 0.0083, and fewer than about 120 surrogates would make every
    # pair uncallable no matter how strong the coupling.
    n_samples = 16000
    n_surrogates = 200 if quick else 400
    seeds = range(3 if quick else 10)

    outcomes = []
    for seed in seeds:
        print(f"realisation {seed}: {n_surrogates} surrogates ...")
        found = surrogate_test(
            simulate(n_samples, seed),
            n_surrogates=n_surrogates,
            seed=seed,
            keep_null=seed == 0,
            time_reversal=True,
        )
        outcomes.append(
            (
                found.dPTE[0, 1],
                bool(found.significant[0, 1]),
                bool(found.directed[0, 2]),
                bool(found.reversal_confirmed[0, 2]),
                bool(found.reversal_confirmed[0, 1]),
            )
        )
        if seed == 0:
            result = found
            print(" ", result.summary())
    (
        artefact,
        artefact_kept,
        coupling_found,
        coupling_confirmed,
        artefact_confirmed,
    ) = map(np.array, zip(*outcomes, strict=True))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)

    plot_null(
        axes[0, 0],
        result,
        0,
        2,
        title="A -> C : real coupling",
    )
    plot_null(
        axes[0, 1],
        result,
        0,
        1,
        title="A -> B : nothing but an SNR gap",
    )

    # -- Benjamini-Hochberg, drawn ----------------------------------------
    off_diagonal = ~np.eye(3, dtype=bool)
    p_flat = np.sort(result.p_values[off_diagonal])
    ranks = np.arange(1, p_flat.size + 1)
    ax = axes[1, 0]
    ax.plot(ranks, p_flat, "o-", color=style.BLUE, ms=5, label="sorted p-values")
    ax.plot(
        ranks,
        0.05 * ranks / p_flat.size,
        "--",
        color=style.OBS,
        label="BH threshold  0.05 k / n",
    )
    ax.set(
        xlabel="rank k",
        ylabel="p-value",
        title="Benjamini-Hochberg: keep everything left of the crossing",
        yscale="log",
    )
    ax.legend(fontsize=8, loc="lower right")

    # -- what the correction removes --------------------------------------
    masked = np.where(result.reversal_confirmed, result.dPTE, np.nan)
    plot_matrix(
        axes[1, 1],
        masked,
        title="dPTE, keeping what time reversal confirms",
        labels=["A", "B", "C"],
        center_on_half=True,
        mask_diagonal=True,
    )
    axes[1, 1].set_facecolor("#f1f3f5")

    save(fig, FIGURES / "significance_testing.png")

    print("\nrealisation 0, raw dPTE (both A->B and A->C look like findings)")
    for i, name in enumerate(CHANNELS):
        row = "  ".join(f"{result.dPTE[i, j]:.3f}" for j in range(3))
        print(f"  {name:<24} {row}")

    print("\nafter testing against the null")
    for source, target, label in ((0, 2, "A -> C (real)"), (0, 1, "A -> B (artefact)")):
        print(
            f"  {label:<20} dPTE {result.dPTE[source, target]:.3f}"
            f"   null mean {result.null_mean[source, target]:.3f}"
            f"   p {result.p_values[source, target]:.3f}"
            f"   {'kept' if result.significant[source, target] else 'rejected'}"
            f", {'' if result.reversal_confirmed[source, target] else 'not '}"
            "confirmed by time reversal"
        )

    n = len(outcomes)
    print(f"\nover {n} realisations")
    print(
        f"  A -> B artefact   mean dPTE {artefact.mean():.3f}, "
        f"kept {artefact_kept.sum()}, confirmed by time reversal "
        f"{artefact_confirmed.sum()}"
    )
    print(
        f"  A -> C coupling   found {coupling_found.sum()}, "
        f"confirmed by time reversal {coupling_confirmed.sum()}"
    )

    assert artefact.mean() > 0.6, "the artefact should look like a finding"
    assert artefact_kept.sum() <= n // 3, "the SNR artefact should mostly be rejected"
    assert coupling_found.all(), "the real coupling must survive, the right way round"
    assert coupling_confirmed.sum() >= n - 1, "time reversal should confirm it"
    assert not artefact_confirmed.any(), "time reversal should never confirm the gap"

    print(
        "\nverified: both pairs look like findings in the raw matrix. The null"
        "\n          rejects the artefact most of the time; time reversal"
        "\n          rejects it every time and confirms the real coupling."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer surrogates")
    main(**vars(parser.parse_args()))
