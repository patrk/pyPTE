"""What a significance test actually does to a dPTE matrix.

Three channels: A really drives C with a lag, B is independent of everything but
much noisier. In the raw matrix both A -> C and A -> B look like findings, and
the artefact is the stronger of the two. Against their own nulls they separate
cleanly, because B's noise raises its surrogate distribution just as much as it
raises the observed value.

Run with:
    uv run python -m examples.significance_testing [--quick]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.utils.plotting import plot_matrix, save, use_house_style
from pyPTE import surrogate_test

FS = 250.0
CHANNELS = ["A (driver)", "B (noisy, independent)", "C (driven by A)"]
FIGURES = Path(__file__).resolve().parent / "figures"


def oscillator(rng: np.random.Generator, n: int, freq: float = 10.0) -> np.ndarray:
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift)


def simulate(n_samples: int, seed: int = 3) -> np.ndarray:
    """A -> C is real. B is independent, and noisy enough to look coupled."""
    rng = np.random.default_rng(seed)

    driver = oscillator(rng, n_samples) + 0.3 * rng.standard_normal(n_samples)
    # four times the noise of A, and no relationship to it whatsoever
    noisy = oscillator(rng, n_samples) + 1.2 * rng.standard_normal(n_samples)
    driven = 0.9 * np.roll(driver, 12) + 0.3 * rng.standard_normal(n_samples)

    return np.vstack([driver, noisy, driven])


def plot_null(ax, result, i: int, j: int, *, title: str) -> None:
    """Where the observation falls inside its own surrogate distribution."""
    null = result.null_distribution[:, i, j]
    observed = result.dPTE[i, j]
    p = result.p_values[i, j]
    survives = bool(result.significant[i, j])

    colour = "#2f6f9f" if survives else "#c2453f"
    ax.hist(null, bins=20, color="#adb5bd", label="surrogate null")
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
        0.98,
        0.95,
        f"p = {p:.3f}\n{'SIGNIFICANT' if survives else 'not significant'}",
        transform=ax.transAxes,
        ha="right",
        va="top",
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

    print(f"running {n_surrogates} surrogates ...")
    result = surrogate_test(
        simulate(n_samples), n_surrogates=n_surrogates, seed=0, keep_null=True
    )
    print(" ", result.summary())

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
    ax.plot(ranks, p_flat, "o-", color="#2f6f9f", ms=5, label="sorted p-values")
    ax.plot(
        ranks,
        0.05 * ranks / p_flat.size,
        "--",
        color="#c2453f",
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
    masked = np.where(result.significant, result.dPTE, np.nan)
    plot_matrix(
        axes[1, 1],
        masked,
        title="dPTE, keeping only what survived",
        labels=["A", "B", "C"],
        center_on_half=True,
        mask_diagonal=True,
    )
    axes[1, 1].set_facecolor("#f1f3f5")

    fig.suptitle("A surrogate test separates real coupling from an SNR artefact")
    save(fig, FIGURES / "significance_testing.png")

    print("\nraw dPTE (both A->B and A->C look like findings)")
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
        )

    assert result.dPTE[0, 1] > 0.6, "the artefact should look like a finding"
    assert result.significant[0, 2], "the real coupling must survive"
    assert not result.significant[0, 1], "the SNR artefact must not survive"

    print(
        "\nverified: the two pairs are indistinguishable in the raw matrix and"
        "\n          cleanly separated against the null. Report dPTE without a"
        "\n          significance test and the artefact is indistinguishable"
        "\n          from the finding."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer surrogates")
    main(**vars(parser.parse_args()))
