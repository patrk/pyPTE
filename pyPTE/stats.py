"""Significance testing for phase transfer entropy.

A raw dPTE value cannot be read as evidence of directed coupling. Two entirely
independent signals produce dPTE well away from 0.5 whenever they differ in
signal-to-noise ratio, because the noisier channel is less predictable from its
own past and the cleaner one therefore looks like a driver. Channel-to-channel
SNR differences are the norm in real recordings, so the question is never "is
this value above 0.5" but "is it further from 0.5 than chance allows".

The null distribution here comes from time-shifted surrogates. Circularly
shifting each channel by an independent random offset leaves every marginal
property of that channel untouched - its amplitude distribution, spectrum and
entropy are exactly preserved - while destroying the cross-channel timing that
directed coupling depends on. Any apparent direction that survives is therefore
a property of the individual channels rather than of their interaction, which
is precisely the bias that needs subtracting.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats as _scipy_stats

from pyPTE.core.pyPTE import PTE

__all__ = [
    "GroupResult",
    "SignificanceResult",
    "benjamini_hochberg",
    "group_contrast",
    "group_test",
    "surrogate_test",
]


@dataclass
class SignificanceResult:
    """Outcome of a surrogate test over every ordered channel pair.

    Attributes
    ----------
    dPTE : numpy.ndarray
        m x m observed dPTE matrix
    raw_PTE : numpy.ndarray
        m x m observed raw PTE matrix, in bits
    p_values : numpy.ndarray
        m x m one-sided p-values against the surrogate null. The diagonal is
        set to 1.0, since self-transfer is identically zero.
    significant : numpy.ndarray
        m x m boolean mask of pairs surviving multiple-comparison correction
    threshold : float
        largest p-value called significant, or 0.0 if nothing survived
    null_mean : numpy.ndarray
        m x m mean of the surrogate null, useful for seeing which pairs carry
        a bias that has nothing to do with coupling
    n_surrogates : int
        number of surrogates the null was built from
    null_distribution : numpy.ndarray | None
        n_surrogates x m x m array of surrogate dPTE values, present only when
        the test was run with ``keep_null=True``. Useful for plotting where an
        observation falls within its own null, at the cost of holding every
        surrogate matrix in memory.
    """

    dPTE: npt.NDArray
    raw_PTE: npt.NDArray
    p_values: npt.NDArray
    significant: npt.NDArray
    threshold: float
    null_mean: npt.NDArray
    n_surrogates: int
    null_distribution: npt.NDArray | None = None

    @property
    def n_significant(self) -> int:
        return int(self.significant.sum())

    def summary(self) -> str:
        m = self.dPTE.shape[0]
        pairs = m * (m - 1)
        return (
            f"{self.n_significant} of {pairs} directed pairs significant "
            f"at p <= {self.threshold:.4g} "
            f"({self.n_surrogates} surrogates, Benjamini-Hochberg corrected)"
        )


def time_shifted_surrogate(
    time_series: npt.NDArray, rng: np.random.Generator, *, min_fraction: float = 0.05
) -> npt.NDArray:
    """Circularly shift every channel by an independent random offset.

    Parameters
    ----------
    time_series : numpy.ndarray
        m x n array of m channels
    rng : numpy.random.Generator
    min_fraction : float
        smallest permitted shift, as a fraction of the recording length. Shifts
        close to zero would leave the original alignment largely intact and bias
        the null towards the observed value.

    Returns
    -------
    surrogate : numpy.ndarray
        m x n array with identical per-channel content and destroyed
        cross-channel timing
    """
    m, n = time_series.shape
    low = max(int(n * min_fraction), 1)
    high = max(n - low, low + 1)
    shifts = rng.integers(low, high, size=m)

    surrogate = np.empty_like(time_series)
    for channel in range(m):
        surrogate[channel] = np.roll(time_series[channel], shifts[channel])
    return surrogate


def benjamini_hochberg(
    p_values: npt.NDArray, alpha: float = 0.05
) -> tuple[npt.NDArray, float]:
    """Control the false discovery rate across a family of p-values.

    Bonferroni is the wrong instrument here: a 100-channel recording yields 9900
    ordered pairs, and dividing alpha by that leaves no power to detect anything.
    Benjamini-Hochberg instead bounds the expected share of false positives
    among the pairs actually called significant.

    Parameters
    ----------
    p_values : numpy.ndarray
        array of p-values, of any shape
    alpha : float
        target false discovery rate

    Returns
    -------
    (significant, threshold) : tuple of (numpy.ndarray, float)
        boolean mask matching the input shape, and the largest p-value called
        significant, which is 0.0 when nothing survives
    """
    flat = np.asarray(p_values, dtype=float).ravel()
    n = flat.size
    order = np.argsort(flat, kind="mergesort")
    ranked = flat[order]

    # the largest k for which p_(k) <= alpha * k / n sets the cutoff
    passes = ranked <= alpha * np.arange(1, n + 1) / n
    if not passes.any():
        return np.zeros_like(p_values, dtype=bool), 0.0

    threshold = float(ranked[np.flatnonzero(passes)[-1]])
    return np.asarray(p_values, dtype=float) <= threshold, threshold


def surrogate_test(
    time_series: npt.ArrayLike,
    *,
    n_surrogates: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    progress: bool = False,
    keep_null: bool = False,
) -> SignificanceResult:
    """Test every directed channel pair against a time-shifted surrogate null.

    Significance is two-tailed in spirit but computed one-sided on the distance
    from 0.5, because dPTE is antisymmetric: evidence that i drives j is the
    same evidence that j does not drive i, and testing |dPTE - 0.5| keeps the
    two directions from being counted as independent findings.

    Parameters
    ----------
    time_series : numpy.ndarray
        m x n array : m channels, n samples
    n_surrogates : int
        size of the null distribution. 200 supports p-values down to ~0.005;
        raise it if you need finer resolution after correction.
    alpha : float
        target false discovery rate for the Benjamini-Hochberg step
    seed : int | None
        seed for surrogate generation, for reproducible results
    progress : bool
        print progress, since a large montage takes a while
    keep_null : bool
        retain every surrogate matrix on the result. Needed to plot where an
        observation sits within its null; costs n_surrogates * m * m floats.

    Returns
    -------
    result : SignificanceResult
    """
    observed = np.asarray(time_series, dtype=float)
    if observed.ndim != 2:
        raise ValueError(f"expected an m x n array, got shape {observed.shape}")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be at least 1")

    rng = np.random.default_rng(seed)
    dPTE, raw_PTE = PTE(observed)

    observed_effect = np.abs(dPTE - 0.5)
    at_least_as_extreme = np.zeros_like(dPTE, dtype=int)
    null_total = np.zeros_like(dPTE, dtype=float)
    null_distribution = (
        np.empty((n_surrogates, *dPTE.shape), dtype=float) if keep_null else None
    )

    for index in range(n_surrogates):
        surrogate_dPTE, _ = PTE(time_shifted_surrogate(observed, rng))
        surrogate_effect = np.abs(surrogate_dPTE - 0.5)
        at_least_as_extreme += surrogate_effect >= observed_effect
        null_total += surrogate_dPTE
        if null_distribution is not None:
            null_distribution[index] = surrogate_dPTE
        if progress and (index + 1) % 25 == 0:
            print(f"  surrogate {index + 1}/{n_surrogates}")

    # the +1 in both terms keeps p strictly positive: with a finite null we can
    # never claim a p-value smaller than 1 / (n_surrogates + 1)
    p_values = (at_least_as_extreme + 1.0) / (n_surrogates + 1.0)
    np.fill_diagonal(p_values, 1.0)

    significant, threshold = benjamini_hochberg(p_values, alpha)
    np.fill_diagonal(significant, False)

    return SignificanceResult(
        dPTE=dPTE,
        raw_PTE=raw_PTE,
        p_values=p_values,
        significant=significant,
        threshold=threshold,
        null_mean=null_total / n_surrogates,
        n_surrogates=n_surrogates,
        null_distribution=null_distribution,
    )


@dataclass
class GroupResult:
    """Outcome of a test across repeated observations of the same channel pairs.

    Attributes
    ----------
    statistic : numpy.ndarray
        m x m test statistic per ordered pair
    p_values : numpy.ndarray
        m x m p-values, with the diagonal set to 1.0
    significant : numpy.ndarray
        m x m boolean mask surviving Benjamini-Hochberg correction
    threshold : float
        largest p-value called significant, or 0.0 if nothing survived
    effect : numpy.ndarray
        m x m mean of the quantity tested: the mean dPTE for group_test, or the
        mean difference between conditions for group_contrast
    n_observations : int
        number of epochs, trials or subjects the test ran over
    method : str
        which test was applied
    """

    statistic: npt.NDArray
    p_values: npt.NDArray
    significant: npt.NDArray
    threshold: float
    effect: npt.NDArray
    n_observations: int
    method: str

    @property
    def n_significant(self) -> int:
        return int(self.significant.sum())

    def summary(self) -> str:
        m = self.statistic.shape[0]
        return (
            f"{self.n_significant} of {m * (m - 1)} directed pairs significant "
            f"at p <= {self.threshold:.4g} ({self.n_observations} observations, "
            f"{self.method}, Benjamini-Hochberg corrected)"
        )


def _pairwise_test(
    samples: npt.NDArray, method: str, alternative: str = "two-sided"
) -> tuple[npt.NDArray, npt.NDArray]:
    """Apply a one-sample test independently to every ordered channel pair."""
    n_obs, m, _ = samples.shape
    statistic = np.zeros((m, m))
    p_values = np.ones((m, m))

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            column = samples[:, i, j]
            if np.allclose(column, column[0]):
                # a constant column carries no evidence either way
                continue
            if method == "wilcoxon":
                result = _scipy_stats.wilcoxon(column, alternative=alternative)
            elif method == "ttest":
                result = _scipy_stats.ttest_1samp(column, 0.0, alternative=alternative)
            else:
                raise ValueError(f"unknown method {method!r}")
            statistic[i, j] = float(result.statistic)
            p_values[i, j] = float(result.pvalue)

    return statistic, p_values


def group_test(
    matrices: npt.ArrayLike,
    *,
    reference: float = 0.5,
    method: Literal["wilcoxon", "ttest"] = "wilcoxon",
    alpha: float = 0.05,
) -> GroupResult:
    """Test whether dPTE departs from chance consistently across observations.

    This is the counterpart to `surrogate_test` for the way M/EEG data usually
    arrives: many short epochs, trials or subjects rather than one long
    recording. It is also far more powerful, because it asks whether an effect
    is *consistent* rather than whether one recording's value is extreme.
    Eighty epochs of a third of a second each can settle a question that a
    minute of continuous data leaves marginal.

    A single epoch may be far too short for a reliable dPTE estimate on its
    own; what matters is that the estimate is unbiased, so the noise averages
    out over observations.

    Parameters
    ----------
    matrices : numpy.ndarray
        n_observations x m x m array of dPTE matrices, one per epoch, trial or
        subject
    reference : float
        the no-effect value to test against. 0.5 for dPTE; use 0.0 when passing
        raw PTE.
    method : {"wilcoxon", "ttest"}
        Wilcoxon signed-rank by default, which makes no normality assumption
        and is the usual choice in the M/EEG literature
    alpha : float
        target false discovery rate

    Returns
    -------
    result : GroupResult

    Notes
    -----
    dPTE is antisymmetric, so ``[i, j]`` and ``[j, i]`` express the same
    finding twice. The correction below treats them as separate tests, which is
    conservative rather than wrong.
    """
    samples = np.asarray(matrices, dtype=float)
    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValueError(f"expected n_observations x m x m, got shape {samples.shape}")
    if samples.shape[0] < 2:
        raise ValueError("a group test needs at least 2 observations")

    centred = samples - reference
    statistic, p_values = _pairwise_test(centred, method)
    significant, threshold = benjamini_hochberg(p_values, alpha)
    np.fill_diagonal(significant, False)

    return GroupResult(
        statistic=statistic,
        p_values=p_values,
        significant=significant,
        threshold=threshold,
        effect=samples.mean(axis=0),
        n_observations=samples.shape[0],
        method=f"{method} vs {reference}",
    )


def group_contrast(
    condition_a: npt.ArrayLike,
    condition_b: npt.ArrayLike,
    *,
    method: Literal["wilcoxon", "ttest"] = "wilcoxon",
    alpha: float = 0.05,
) -> GroupResult:
    """Test whether directed connectivity differs between two conditions.

    The paired comparison most connectivity studies actually want: the same
    subjects measured twice, asking which pairs changed. Both inputs must be
    ordered identically, so that row k of each is the same subject.

    Parameters
    ----------
    condition_a, condition_b : numpy.ndarray
        n_observations x m x m dPTE matrices, paired along the first axis
    method : {"wilcoxon", "ttest"}
        Wilcoxon signed-rank by default
    alpha : float
        target false discovery rate

    Returns
    -------
    result : GroupResult
        ``effect`` holds the mean difference, condition_a minus condition_b
    """
    a = np.asarray(condition_a, dtype=float)
    b = np.asarray(condition_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"conditions must have matching shapes, got {a.shape} {b.shape}"
        )
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"expected n_observations x m x m, got shape {a.shape}")
    if a.shape[0] < 2:
        raise ValueError("a paired contrast needs at least 2 observations")

    difference = a - b
    statistic, p_values = _pairwise_test(difference, method)
    significant, threshold = benjamini_hochberg(p_values, alpha)
    np.fill_diagonal(significant, False)

    return GroupResult(
        statistic=statistic,
        p_values=p_values,
        significant=significant,
        threshold=threshold,
        effect=difference.mean(axis=0),
        n_observations=a.shape[0],
        method=f"paired {method}",
    )
