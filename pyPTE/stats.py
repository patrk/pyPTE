"""Significance testing for phase transfer entropy.

A raw dPTE value is not evidence of coupling on its own. Two independent signals
land well away from 0.5 whenever they differ in signal-to-noise ratio, because
the noisier one is harder to predict from its own past, so the cleaner one looks
like a driver. Real channels always differ in SNR, so the question is not "is
this above 0.5" but "is it further from 0.5 than chance allows".

The null comes from time-shifted surrogates: circularly shifting each channel by
its own random offset preserves that channel's spectrum, amplitude distribution
and entropy exactly, while destroying the cross-channel timing that coupling
depends on. Every two channels are shifted at least 5% of the recording apart,
since a pair moved by nearly the same offset keeps its timing. Whatever
survives is a property of the interaction rather than of the individual
channels.
"""

import os
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product, repeat
from typing import Literal, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt
from scipy import stats as _scipy_stats

from pyPTE.core.pyPTE import (
    BinningRule,
    DelayRule,
    compute_dPTE_rawPTE,
    prepare_phase,
    reversal_differenced_dPTE,
    reverse_phase,
)

__all__ = [
    "ClusterResult",
    "GroupResult",
    "SignificanceResult",
    "benjamini_hochberg",
    "cluster_permutation",
    "group_contrast",
    "group_test",
    "surrogate_test",
    "surrogates_needed",
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
    effective_PTE : numpy.ndarray
        m x m raw PTE minus the mean raw PTE of the surrogates, in bits: the
        effective transfer entropy of Marschinski & Kantz (2002) computed on
        phase, as in Yang et al. (2017) and Kudo et al. (2022), but against
        time-shifted rather than shuffled surrogates. The surrogates carry
        the estimator's bias but no coupling, so the difference is the
        transfer beyond chance, comparable across recordings and bin counts
        where raw PTE is not. Values below zero are kept, since clipping them
        would bias any average taken across subjects. The diagonal is zero.
    p_values : numpy.ndarray
        m x m one-sided p-values against the surrogate null. The diagonal is
        set to 1.0, since self-transfer is identically zero.
    significant : numpy.ndarray
        m x m boolean mask of pairs surviving multiple-comparison correction.
        It marks a pair in both orientations, since ``dPTE[j, i]`` is
        ``1 - dPTE[i, j]`` and the test is the same; which way the pair points
        is the side of ``centre`` dPTE falls on. Use ``directed`` to score
        against a directed ground truth, or each real edge also counts as a
        false one.
    threshold : float
        largest p-value called significant, or 0.0 if nothing survived
    null_mean : numpy.ndarray
        m x m mean of the surrogate null, useful for seeing which pairs carry
        a bias that has nothing to do with coupling
    n_surrogates : int
        number of surrogates the null was built from
    raw_null_mean : numpy.ndarray
        m x m mean raw PTE of the surrogates, in bits: the bias floor that
        effective_PTE subtracts
    null_distribution : numpy.ndarray | None
        n_surrogates x m x m array of surrogate dPTE values, present only when
        the test was run with ``keep_null=True``. Useful for plotting where an
        observation falls within its own null, at the cost of holding every
        surrogate matrix in memory.
    reversed_dPTE : numpy.ndarray | None
        m x m dPTE of the time-reversed recording, present only when the test
        was run with ``time_reversal=True``
    reversed_p_values : numpy.ndarray | None
        m x m p-values of the time-reversed recording against its own
        surrogates, built from the same shifts
    reversal_confirmed : numpy.ndarray | None
        m x m boolean mask of significant pairs whose direction flips,
        significantly, when time is reversed. Like ``significant`` it is
        symmetric: the mask marks a pair, and dPTE says which way it points.
    differenced_dPTE : numpy.ndarray | None
        m x m time-reversal-differenced dPTE (see
        pyPTE.core.pyPTE.reversal_differenced_dPTE), in which directions that
        come from a difference in signal-to-noise ratio or from instantaneous
        mixing cancel. It is reported as an effect size only. The surrogate
        null assumes independent channels, which mixing violates, so to test
        it, compute it per segment or epoch (`PTE_segments`, or
        ``PTE_from_epochs(time_reversal=True)``) and use `group_test`.
    centre : numpy.ndarray | float
        what the test measured distance from: 0.5, or with
        ``centre_on_null=True`` the m x m median of each pair's surrogate null.
        A pair points from i to j when ``dPTE[i, j]`` is above it.
    """

    dPTE: npt.NDArray[np.float64]
    raw_PTE: npt.NDArray[np.float64]
    effective_PTE: npt.NDArray[np.float64]
    p_values: npt.NDArray[np.float64]
    significant: npt.NDArray[np.bool_]
    threshold: float
    null_mean: npt.NDArray[np.float64]
    n_surrogates: int
    raw_null_mean: npt.NDArray[np.float64]
    null_distribution: npt.NDArray[np.float64] | None = None
    reversed_dPTE: npt.NDArray[np.float64] | None = None
    reversed_p_values: npt.NDArray[np.float64] | None = None
    reversal_confirmed: npt.NDArray[np.bool_] | None = None
    differenced_dPTE: npt.NDArray[np.float64] | None = None
    centre: npt.NDArray[np.float64] | float = 0.5

    @property
    def n_significant(self) -> int:
        return int(self.significant.sum())

    @property
    def directed(self) -> npt.NDArray[np.bool_]:
        """m x m mask of significant pairs, at [driver, target] only."""
        return self.significant & (self.dPTE > self.centre)

    def summary(self) -> str:
        m = self.dPTE.shape[0]
        pairs = m * (m - 1) // 2
        reversal = (
            ""
            if self.reversal_confirmed is None
            else f", {int(self.reversal_confirmed.sum()) // 2} of these confirmed "
            "by time reversal"
        )
        return (
            f"{int(self.directed.sum())} of {pairs} channel pairs significant "
            f"at p <= {self.threshold:.4g}{reversal} "
            f"({self.n_surrogates} surrogates, Benjamini-Hochberg corrected)"
        )


_Scalar = TypeVar("_Scalar", bound=np.generic)


def time_shifted_surrogate(
    time_series: npt.NDArray[_Scalar],
    rng: np.random.Generator,
    *,
    min_fraction: float = 0.05,
) -> npt.NDArray[_Scalar]:
    """Circularly shift every channel by its own random offset.

    Parameters
    ----------
    time_series : numpy.ndarray
        m x n array of m channels
    rng : numpy.random.Generator
    min_fraction : float
        smallest permitted shift of any channel relative to any other, as a
        fraction of the recording length. Two channels shifted by nearly the
        same offset keep their alignment, and with it their coupling, which
        biases the null towards the observed value. With more channels than
        fit at that spacing, they are spread evenly instead.

    Returns
    -------
    surrogate : numpy.ndarray
        m x n array with identical per-channel content and destroyed
        cross-channel timing
    """
    m, n = time_series.shape
    gap = _shift_gap(m, n, min_fraction)
    return _apply_shifts(time_series, _draw_shifts(m, n, rng, gap))


def _shift_gap(m: int, n: int, min_fraction: float = 0.05) -> int:
    """Smallest shift between two channels, as _draw_shifts keeps them apart."""
    gap = max(int(n * min_fraction), 1)
    if m * gap > n:
        gap = max(n // m, 1)
        warnings.warn(
            f"too many channels to shift every pair at least {min_fraction:.0%} "
            f"of the recording apart; pairs are kept {gap} samples apart instead",
            RuntimeWarning,
            stacklevel=3,
        )
    return gap


def _draw_shifts(
    m: int, n: int, rng: np.random.Generator, gap: int
) -> npt.NDArray[np.int64]:
    """One random circular shift per channel, every two at least gap apart.

    Only shifts relative to each other matter for a pair, so the m shifts are
    placed on the circle of n samples with at least gap between neighbours:
    m sorted draws from the n - m * gap samples left over, each moved on by
    gap times its rank, then rotated at random and dealt to random channels.
    """
    slack = rng.integers(0, n - m * gap + 1, size=m)
    points = np.sort(slack) + gap * np.arange(m)
    return rng.permutation((points + rng.integers(n)) % n)


def _apply_shifts(
    time_series: npt.NDArray[_Scalar], shifts: npt.NDArray[np.int64]
) -> npt.NDArray[_Scalar]:
    surrogate = np.empty_like(time_series)
    for channel in range(time_series.shape[0]):
        surrogate[channel] = np.roll(time_series[channel], shifts[channel])
    return surrogate


_Block = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None
]


def _surrogate_block(
    d_phase: npt.NDArray[np.int32],
    delay: int,
    n_bins: int,
    shifts: npt.NDArray[np.int64],
    reverse: bool = False,
) -> _Block:
    """dPTE and raw PTE for one surrogate per row of shifts; a worker task.

    With reverse, also the dPTE of each surrogate with time reversed, which is
    the matching surrogate of the reversed recording.
    """
    m = d_phase.shape[0]
    dPTE = np.empty((len(shifts), m, m))
    raw_PTE = np.empty((len(shifts), m, m))
    reversed_dPTE = np.empty((len(shifts), m, m)) if reverse else None
    for row, channel_shifts in enumerate(shifts):
        surrogate = _apply_shifts(d_phase, channel_shifts)
        dPTE[row], raw_PTE[row] = compute_dPTE_rawPTE(surrogate, delay, n_bins=n_bins)
        if reversed_dPTE is not None:
            reversed_dPTE[row], _ = compute_dPTE_rawPTE(
                reverse_phase(surrogate), delay, n_bins=n_bins
            )
    return dPTE, raw_PTE, reversed_dPTE


def _worker_count(n_jobs: int) -> int:
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive count or -1, got {n_jobs}")
    return n_jobs


def surrogates_needed(n_channels: int, alpha: float = 0.05) -> int:
    """Smallest surrogate count that could yield a single significant edge.

    A surrogate test can never report a p-value below ``1 / (n_surrogates + 1)``.
    Its p-values are symmetric, ``p[i, j] == p[j, i]``, so an isolated pair
    enters the correction twice and Benjamini-Hochberg accepts it at rank 2 once
    p falls below ``alpha / n_pairs``, counting unordered pairs. Below the count
    returned here, no edge can survive no matter how strong the coupling.

    This is a worst case. A genuinely widespread effect is accepted at a higher
    rank, where the threshold is looser, so fewer surrogates can still work when
    many pairs are truly coupled.

    Parameters
    ----------
    n_channels : int
    alpha : float

    Returns
    -------
    n_surrogates : int
    """
    n_pairs = n_channels * (n_channels - 1) // 2
    if n_pairs < 1:
        return 1
    return int(np.ceil(n_pairs / alpha)) - 1


def warn_if_underpowered(
    n_channels: int, n_surrogates: int, alpha: float = 0.05
) -> None:
    """Warn when the surrogate count cannot produce a significant result."""
    n_pairs = n_channels * (n_channels - 1) // 2
    if n_pairs < 1:
        return

    floor = 1.0 / (n_surrogates + 1)
    strictest = alpha / n_pairs
    if floor <= strictest:
        return

    warnings.warn(
        f"{n_surrogates} surrogates cannot produce a p-value below {floor:.3g}, "
        f"but Benjamini-Hochberg over {n_pairs} channel pairs needs "
        f"{strictest:.3g} to "
        f"accept an isolated edge. Unless the effect is widespread, expect "
        f"nothing to survive. Use at least "
        f"{surrogates_needed(n_channels, alpha)} surrogates, or test at the "
        f"cluster level with cluster_permutation().",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_if_too_few_observations(
    centred: npt.NDArray[np.float64], alpha: float
) -> None:
    """Warn when Wilcoxon over this many observations cannot reach significance.

    The smallest two-sided p-value the exact test can return for n
    observations is 2 / 2**n, when every observation has the same sign.
    """
    n_obs, m, _ = centred.shape
    n_tests = m * (m - 1)
    if _is_antisymmetric(centred):
        # mirrored, so one test per unordered pair
        n_tests //= 2
    if n_tests < 1:
        return
    floor = 2.0 / 2.0**n_obs
    strictest = alpha / n_tests
    if floor <= strictest:
        return
    needed = int(np.ceil(np.log2(2.0 * n_tests / alpha)))
    warnings.warn(
        f"Wilcoxon over {n_obs} observations cannot produce a p-value below "
        f"{floor:.3g}, but Benjamini-Hochberg over {n_tests} "
        f"test{'s' if n_tests != 1 else ''} needs "
        f"{strictest:.3g} to accept an isolated edge. Unless the effect is "
        f"widespread, expect nothing to survive. Use at least {needed} epochs "
        f"or segments, or test at the cluster level with cluster_permutation().",
        RuntimeWarning,
        stacklevel=3,
    )


def benjamini_hochberg(
    p_values: npt.NDArray[np.float64], alpha: float = 0.05
) -> tuple[npt.NDArray[np.bool_], float]:
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


def _correct_off_diagonal(
    p_values: npt.NDArray[np.float64], alpha: float
) -> tuple[npt.NDArray[np.bool_], float]:
    """Benjamini-Hochberg over the off-diagonal entries of an m x m matrix.

    The diagonal is self-transfer, never a test, so letting it into the family
    would inflate n from m(m-1) to m^2 and tighten every threshold for nothing.
    """
    p_values = np.asarray(p_values, dtype=float)
    off_diagonal = ~np.eye(p_values.shape[0], dtype=bool)
    significant = np.zeros_like(p_values, dtype=bool)
    significant[off_diagonal], threshold = benjamini_hochberg(
        p_values[off_diagonal], alpha
    )
    return significant, threshold


class _ShiftTest(NamedTuple):
    dPTE: npt.NDArray[np.float64]
    raw_PTE: npt.NDArray[np.float64]
    p_values: npt.NDArray[np.float64]
    null_mean: npt.NDArray[np.float64]
    raw_null_mean: npt.NDArray[np.float64]
    null_distribution: npt.NDArray[np.float64] | None
    reversed_dPTE: npt.NDArray[np.float64] | None = None
    reversed_p_values: npt.NDArray[np.float64] | None = None
    centre: npt.NDArray[np.float64] | float = 0.5
    reversed_centre: npt.NDArray[np.float64] | float = 0.5


def _p_values(
    extreme: npt.NDArray[np.int_],
    n_surrogates: int,
    undefined: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    # the +1 in both terms keeps p strictly positive: with a finite null we can
    # never claim a p-value smaller than 1 / (n_surrogates + 1)
    p_values = ((extreme + 1.0) / (n_surrogates + 1.0)).astype(np.float64)
    p_values[undefined] = 1.0
    np.fill_diagonal(p_values, 1.0)
    return p_values


def _count_extreme(
    surrogate: npt.NDArray[np.float64], observed_effect: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    effect = np.abs(surrogate - 0.5)
    # an undefined surrogate is counted as extreme, the conservative side
    extreme: npt.NDArray[np.bool_] = (effect >= observed_effect) | np.isnan(effect)
    return extreme


def _centred_count(
    null: npt.NDArray[np.float64], observed: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Count surrogates as far from the null median as the observation.

    Returns the counts and the per-pair median. dPTE[j, i] is 1 - dPTE[i, j]
    in every surrogate too, so the medians mirror and a pair keeps one test.
    """
    with warnings.catch_warnings():
        # a pair whose every surrogate is undefined has no median
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(null, axis=0)
    centre[np.isnan(centre)] = 0.5
    np.fill_diagonal(centre, 0.5)
    effect = np.abs(null - centre)
    extreme = (effect >= np.abs(observed - centre)) | np.isnan(effect)
    return extreme.sum(axis=0), centre


def _shift_test(
    d_phase: npt.NDArray[np.int32],
    analysis_delay: int,
    n_bins: int,
    shifts: npt.NDArray[np.int64],
    workers: int,
    keep_null: bool,
    progress: bool,
    reverse: bool = False,
    centre_on_null: bool = False,
) -> _ShiftTest:
    """Observed dPTE and its p-values against the surrogates given by shifts.

    With reverse, also the dPTE of the time-reversed recording, tested
    against the reverse of the same surrogates. With centre_on_null, distance
    is measured from each pair's null median rather than 0.5, which needs the
    whole null in memory before anything can be counted.
    """
    n_surrogates = len(shifts)
    dPTE, raw_PTE = compute_dPTE_rawPTE(d_phase, analysis_delay, n_bins=n_bins)
    undefined = np.isnan(dPTE)

    reversed_dPTE = None
    if reverse:
        reversed_dPTE, _ = compute_dPTE_rawPTE(
            reverse_phase(d_phase), analysis_delay, n_bins=n_bins
        )
        reversed_extreme = np.zeros_like(dPTE, dtype=int)

    observed_effect = np.abs(dPTE - 0.5)
    at_least_as_extreme = np.zeros_like(dPTE, dtype=int)
    null_total = np.zeros_like(dPTE, dtype=float)
    raw_null_total = np.zeros_like(raw_PTE, dtype=float)
    null_distribution = (
        np.empty((n_surrogates, *dPTE.shape), dtype=float)
        if keep_null or centre_on_null
        else None
    )
    reversed_null = (
        np.empty((n_surrogates, *dPTE.shape), dtype=float)
        if reverse and centre_on_null
        else None
    )

    def blocks() -> Iterator[_Block]:
        if workers == 1:
            for row in shifts:
                yield _surrogate_block(
                    d_phase, analysis_delay, n_bins, row[None], reverse
                )
            return
        chunks = np.array_split(shifts, min(4 * workers, n_surrogates))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(
                _surrogate_block,
                repeat(d_phase),
                repeat(analysis_delay),
                repeat(n_bins),
                chunks,
                repeat(reverse),
            )

    index = 0
    for dPTE_block, raw_block, reversed_block in blocks():
        # one surrogate at a time, so the sum does not depend on the chunking
        for row, (surrogate_dPTE, surrogate_raw) in enumerate(
            zip(dPTE_block, raw_block, strict=True)
        ):
            raw_null_total += surrogate_raw
            at_least_as_extreme += _count_extreme(surrogate_dPTE, observed_effect)
            if reversed_block is not None:
                assert reversed_dPTE is not None
                reversed_extreme += _count_extreme(
                    reversed_block[row], np.abs(reversed_dPTE - 0.5)
                )
                if reversed_null is not None:
                    reversed_null[index] = reversed_block[row]
            null_total += surrogate_dPTE
            if null_distribution is not None:
                null_distribution[index] = surrogate_dPTE
            index += 1
            if progress and index % 25 == 0:
                print(f"  surrogate {index}/{n_surrogates}")

    centre: npt.NDArray[np.float64] | float = 0.5
    reversed_centre: npt.NDArray[np.float64] | float = 0.5
    if centre_on_null:
        assert null_distribution is not None
        at_least_as_extreme, centre = _centred_count(null_distribution, dPTE)
        if reversed_null is not None:
            assert reversed_dPTE is not None
            reversed_extreme, reversed_centre = _centred_count(
                reversed_null, reversed_dPTE
            )

    reversed_p = None
    if reversed_dPTE is not None:
        reversed_p = _p_values(
            reversed_extreme, n_surrogates, undefined | np.isnan(reversed_dPTE)
        )

    return _ShiftTest(
        dPTE,
        raw_PTE,
        _p_values(at_least_as_extreme, n_surrogates, undefined),
        null_total / n_surrogates,
        raw_null_total / n_surrogates,
        null_distribution if keep_null else None,
        reversed_dPTE,
        reversed_p,
        centre,
        reversed_centre,
    )


def surrogate_test(
    time_series: npt.ArrayLike,
    *,
    n_surrogates: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    progress: bool = False,
    keep_null: bool = False,
    binning: BinningRule = "scott",
    delay: DelayRule = "zero-crossing",
    trim_cycles: float = 0.0,
    n_jobs: int = 1,
    time_reversal: bool = False,
    centre_on_null: bool = False,
) -> SignificanceResult:
    """Test every directed channel pair against a time-shifted surrogate null.

    Significance is two-tailed in spirit but computed one-sided on the distance
    from 0.5, because dPTE is antisymmetric: evidence that i drives j is the
    same evidence that j does not drive i, and testing ``|dPTE - 0.5|`` keeps the
    two directions from being counted as independent findings.

    The null rules out independent channels, not a wrong direction: a
    difference in signal-to-noise ratio between coupled channels, or one
    source mixed into both, gives a significant direction that may point
    either way. Band-pass narrowly first, and check direction with
    ``time_reversal=True`` or, with more power, with `PTE_segments` and
    `group_test`.

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
    binning, delay, trim_cycles
        as for `PTE`. The delay and bins are estimated once, from the observed
        data, and every surrogate reuses them, so the null is computed with
        exactly the settings of the value it is compared against.
    n_jobs : int
        worker processes for the surrogates; -1 uses every CPU. Results do not
        depend on it: the shifts are drawn up front from ``seed``. Starting
        workers costs about a second, so it pays off on large montages.
    time_reversal : bool
        also test the time-reversed recording (Haufe et al. 2012, Winkler et
        al. 2016). A real direction reverses with time; one produced by a
        difference in signal-to-noise ratio or by instantaneous mixing does
        not. ``reversal_confirmed`` marks pairs that are significant both ways
        round, with opposite signs, which is the conservative rule; the
        reversed recording is tested against the reverse of each surrogate.
        ``differenced_dPTE`` comes back too, as an effect size. It doubles the
        run time, and it cannot catch a common driver that reaches the two
        channels at different lags, since that order is genuinely in the data.
    centre_on_null : bool
        measure each pair's distance from the median of its own surrogate
        null instead of from 0.5, and read its direction from that side too
        (``result.centre``). A difference in signal-to-noise ratio shifts the
        whole null away from 0.5; when it opposes a real coupling, the plain
        test can lose the coupling or call it the wrong way, and centring
        recovers much of it. It does not lower the false-positive rate on
        independent channels, and when the noisier channel is the driver and
        its signal-to-noise ratio is low, the direction can still come out
        wrong: coupled channels show more of the artefact than their shifted
        surrogates do. Combined with ``time_reversal=True``, read
        ``reversal_confirmed``, which removes those wrong directions in
        simulation. It holds every surrogate matrix in memory, as
        ``keep_null`` does.

    Returns
    -------
    result : SignificanceResult
    """
    observed = np.asarray(time_series, dtype=float)
    if observed.ndim != 2:
        raise ValueError(f"expected an m x n array, got shape {observed.shape}")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be at least 1")
    workers = _worker_count(n_jobs)

    warn_if_underpowered(observed.shape[0], n_surrogates, alpha)

    rng = np.random.default_rng(seed)
    d_phase, analysis_delay, n_bins = prepare_phase(
        observed, binning=binning, delay=delay, trim_cycles=trim_cycles
    )
    # shifting the binned phase rather than the raw signal keeps the seam out
    # of the Hilbert transform. Drawing every shift here, in order, makes the
    # null identical however many workers compute it.
    m, n = d_phase.shape
    gap = _shift_gap(m, n)
    shifts = np.array([_draw_shifts(m, n, rng, gap) for _ in range(n_surrogates)])

    forward = _shift_test(
        d_phase,
        analysis_delay,
        n_bins,
        shifts,
        workers,
        keep_null,
        progress,
        reverse=time_reversal,
        centre_on_null=centre_on_null,
    )

    # zero transfer in both directions, e.g. a duplicated or flat channel,
    # makes dPTE 0/0; NaN compares False with everything, which would count
    # no surrogate as extreme and hand the pair the smallest possible p-value,
    # so _shift_test reports these pairs with p = 1
    undefined = np.isnan(forward.dPTE)
    np.fill_diagonal(undefined, False)
    if undefined.any():
        warnings.warn(
            f"dPTE is undefined for {int(undefined.sum()) // 2} channel pair(s) "
            "with no transfer in either direction, such as duplicated or flat "
            "channels; they are reported with p = 1",
            RuntimeWarning,
            stacklevel=2,
        )

    significant, threshold = _correct_off_diagonal(forward.p_values, alpha)

    effective_PTE = forward.raw_PTE - forward.raw_null_mean
    np.fill_diagonal(effective_PTE, 0.0)

    reversal_confirmed = differenced_dPTE = None
    if forward.reversed_p_values is not None and forward.reversed_dPTE is not None:
        reversed_significant, _ = _correct_off_diagonal(
            forward.reversed_p_values, alpha
        )
        flips = np.sign(forward.dPTE - forward.centre) == -np.sign(
            forward.reversed_dPTE - forward.reversed_centre
        )
        reversal_confirmed = significant & reversed_significant & flips
        differenced_dPTE = reversal_differenced_dPTE(
            forward.dPTE, forward.reversed_dPTE
        )

    return SignificanceResult(
        dPTE=forward.dPTE,
        raw_PTE=forward.raw_PTE,
        effective_PTE=effective_PTE,
        p_values=forward.p_values,
        significant=significant,
        threshold=threshold,
        null_mean=forward.null_mean,
        n_surrogates=n_surrogates,
        raw_null_mean=forward.raw_null_mean,
        null_distribution=forward.null_distribution,
        reversed_dPTE=forward.reversed_dPTE,
        reversed_p_values=forward.reversed_p_values,
        reversal_confirmed=reversal_confirmed,
        differenced_dPTE=differenced_dPTE,
        centre=forward.centre,
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
        m x m boolean mask surviving Benjamini-Hochberg correction. For dPTE
        and for differences of dPTE it marks a pair in both orientations, as
        the two are one test; ``effect`` says which way it points (above the
        reference of 0.5 for group_test, above 0 for group_contrast).
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

    statistic: npt.NDArray[np.float64]
    p_values: npt.NDArray[np.float64]
    significant: npt.NDArray[np.bool_]
    threshold: float
    effect: npt.NDArray[np.float64]
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


def _is_antisymmetric(samples: npt.NDArray[np.float64]) -> bool:
    """Whether ``samples[:, j, i] == -samples[:, i, j]`` off the diagonal.

    True for centred dPTE and for differences of dPTE, where the two
    orientations of a pair are one finding; false for raw PTE.
    """
    off_diagonal = ~np.eye(samples.shape[1], dtype=bool)
    mirrored = -samples.transpose(0, 2, 1)
    return bool(
        np.allclose(
            samples[:, off_diagonal],
            mirrored[:, off_diagonal],
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        )
    )


def _pairwise_test(
    samples: npt.NDArray[np.float64], method: str, alternative: str = "two-sided"
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Apply a one-sample test to every channel pair.

    Antisymmetric input is tested once per unordered pair and mirrored: the
    lower entry is the same test on the negated column, which leaves a
    two-sided p-value unchanged and flips the sign of a t-statistic.
    """
    n_obs, m, _ = samples.shape
    statistic = np.zeros((m, m))
    p_values = np.ones((m, m))
    mirror = alternative == "two-sided" and _is_antisymmetric(samples)

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if mirror and j < i:
                # Wilcoxon's two-sided statistic, min(W+, W-), is sign-blind
                sign = -1.0 if method == "ttest" else 1.0
                statistic[i, j] = sign * statistic[j, i]
                p_values[i, j] = p_values[j, i]
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

    The counterpart to `surrogate_test` for repeated observations: epochs,
    trials, segments or subjects. It asks whether an effect is consistent
    across them.

    Consistency is also what an artefact produces. A difference in
    signal-to-noise ratio, or one source mixed into two channels, shifts dPTE
    the same way in every observation, so plain dPTE matrices test for those
    as readily as for coupling. Pass time-reversal-differenced matrices
    (``time_reversal=True`` in `PTE_segments` or ``PTE_from_epochs``), and
    keep their default ``binning="locking"``: with too few bins a real
    coupling's direction depends on the delay and can come out significantly
    backwards. Short observations carry little directional information, so
    the estimate in each one is pulled towards 0.5 rather than merely noisy.

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
    finding. When the input is antisymmetric each pair is tested once and the
    result mirrored. Benjamini-Hochberg over the mirrored p-values, each
    present twice, reaches exactly the decisions it would over one per pair.
    Raw PTE is not antisymmetric, so there both orientations are tested.
    """
    samples = np.asarray(matrices, dtype=float)
    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValueError(f"expected n_observations x m x m, got shape {samples.shape}")
    if samples.shape[0] < 2:
        raise ValueError("a group test needs at least 2 observations")

    centred = samples - reference
    if method == "wilcoxon":
        _warn_if_too_few_observations(centred, alpha)
    statistic, p_values = _pairwise_test(centred, method)
    significant, threshold = _correct_off_diagonal(p_values, alpha)

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
    significant, threshold = _correct_off_diagonal(p_values, alpha)

    return GroupResult(
        statistic=statistic,
        p_values=p_values,
        significant=significant,
        threshold=threshold,
        effect=difference.mean(axis=0),
        n_observations=a.shape[0],
        method=f"paired {method}",
    )


@dataclass
class ClusterResult:
    """Outcome of a cluster-based permutation test over a connectivity matrix.

    Attributes
    ----------
    statistic : numpy.ndarray
        m x m t-statistic per ordered pair
    clusters : list of numpy.ndarray
        boolean m x m masks, one per connected component found above the
        cluster-forming threshold, ordered from strongest to weakest. For dPTE
        each pair appears once, at ``[i, j]`` where i drives j.
    cluster_p : numpy.ndarray
        p-value per cluster, against the null of the largest cluster mass
    significant : numpy.ndarray
        m x m boolean mask: the union of the clusters that survived, marking
        both orientations of each pair for dPTE
    threshold : float
        the cluster-forming t-threshold that was applied
    n_permutations : int
        size of the null that was used, 2 ** n_observations when exact
    alpha : float
        the cluster-level significance threshold that was applied
    """

    statistic: npt.NDArray[np.float64]
    clusters: list[npt.NDArray[np.bool_]]
    cluster_p: npt.NDArray[np.float64]
    significant: npt.NDArray[np.bool_]
    threshold: float
    n_permutations: int
    alpha: float = 0.05

    @property
    def n_significant_clusters(self) -> int:
        return int((self.cluster_p <= self.alpha).sum())

    def summary(self) -> str:
        if not self.clusters:
            return (
                f"no edges passed the cluster-forming threshold "
                f"t = {self.threshold:.2f}"
            )
        best = float(self.cluster_p.min())
        return (
            f"{len(self.clusters)} cluster(s) above t = {self.threshold:.2f}; "
            f"smallest cluster p = {best:.4g} "
            f"({self.n_permutations} permutations, {int(self.significant.sum())} "
            f"edges retained)"
        )


def _edgewise_t(samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """One-sample t-statistic per edge, with zero-variance edges set to zero."""
    n = samples.shape[0]
    mean = samples.mean(axis=0)
    sd = samples.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t: npt.NDArray[np.float64] = mean / (sd / np.sqrt(n))
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _find_clusters(
    statistic: npt.NDArray[np.float64],
    threshold: float,
    signs: tuple[float, ...] = (1.0, -1.0),
) -> tuple[list[npt.NDArray[np.bool_]], npt.NDArray[np.float64]]:
    """Connected components of the supra-threshold edges, and their masses.

    Positive and negative effects are clustered separately, so an increase and
    a decrease can never be joined into one component through a shared node.
    For an antisymmetric statistic the negative clusters mirror the positive
    ones exactly, so passing ``signs=(1.0,)`` finds each pair once.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    m = statistic.shape[0]
    off_diagonal = ~np.eye(m, dtype=bool)

    clusters: list[npt.NDArray[np.bool_]] = []
    masses: list[float] = []

    for sign in signs:
        supra = (statistic * sign > threshold) & off_diagonal
        if not supra.any():
            continue

        # weak connectivity: an edge links its two nodes regardless of direction
        n_components, labels = connected_components(
            csr_matrix(supra), directed=True, connection="weak"
        )
        for component in range(n_components):
            in_component = labels == component
            mask = supra & in_component[:, np.newaxis] & in_component[np.newaxis, :]
            if mask.any():
                clusters.append(mask)
                masses.append(float(np.abs(statistic[mask]).sum()))

    if not clusters:
        return [], np.empty(0)

    order = np.argsort(masses)[::-1]
    return [clusters[k] for k in order], np.asarray(masses)[order]


def cluster_permutation(
    condition_a: npt.ArrayLike,
    condition_b: npt.ArrayLike | None = None,
    *,
    reference: float = 0.5,
    threshold_p: float = 0.01,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> ClusterResult:
    """Cluster-based permutation test over a connectivity matrix.

    The multiple-comparison approach the M/EEG literature expects, adapted to
    connectivity as the network-based statistic: threshold the edgewise
    statistic, group the survivors into connected components, and compare each
    component's mass against the null of the *largest* component under random
    sign flips. Because the null is built from the maximum, family-wise error
    is controlled across the whole matrix without testing each edge separately.

    It is more powerful than Benjamini-Hochberg when an effect is distributed
    over several connected edges, and less powerful when effects are isolated,
    since a lone edge forms a cluster of one. Its claim is also weaker: a
    significant cluster means "this component contains an effect", not "each of
    these edges is an effect".

    Parameters
    ----------
    condition_a : numpy.ndarray
        n_observations x m x m array of dPTE matrices
    condition_b : numpy.ndarray | None
        paired second condition. When omitted, condition_a is tested against
        `reference` instead.
    reference : float
        no-effect value for the one-sample case; 0.5 for dPTE
    threshold_p : float
        cluster-forming threshold, as an uncorrected two-tailed p-value. This
        is a free parameter of the method, not an inference: raising it favours
        few large clusters, lowering it favours many small ones.
    n_permutations : int
        size of the permutation null. When 2 ** n_observations is no larger,
        every sign pattern is used once instead, which makes the test exact.
    alpha : float
        cluster-level significance
    seed : int | None

    Returns
    -------
    result : ClusterResult
    """
    a = np.asarray(condition_a, dtype=float)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"expected n_observations x m x m, got shape {a.shape}")

    if condition_b is None:
        samples = a - reference
    else:
        b = np.asarray(condition_b, dtype=float)
        if b.shape != a.shape:
            raise ValueError(
                f"conditions must have matching shapes, got {a.shape} {b.shape}"
            )
        samples = a - b

    n_obs = samples.shape[0]
    if n_obs < 2:
        raise ValueError("a permutation test needs at least 2 observations")

    threshold = float(_scipy_stats.t.ppf(1.0 - threshold_p / 2.0, df=n_obs - 1))

    # dPTE and its differences are antisymmetric: a cluster of i -> j effects
    # has a mirror-image negative cluster, which is the same finding
    antisymmetric = _is_antisymmetric(samples)
    signs = (1.0,) if antisymmetric else (1.0, -1.0)

    observed = _edgewise_t(samples)
    clusters, masses = _find_clusters(observed, threshold, signs)

    if not clusters:
        return ClusterResult(
            statistic=observed,
            clusters=[],
            cluster_p=np.empty(0),
            significant=np.zeros_like(observed, dtype=bool),
            threshold=threshold,
            n_permutations=n_permutations,
            alpha=alpha,
        )

    # sign flipping is the exchangeability the paired and one-sample designs
    # provide: under the null, the sign of each observation's effect is arbitrary
    exact = 2**n_obs <= n_permutations
    if exact:
        # few enough observations to take every sign pattern once, the observed
        # one included, which gives the exact permutation p-value
        flip_sets = np.array(list(product((1.0, -1.0), repeat=n_obs)))
        n_permutations = len(flip_sets)
    else:
        rng = np.random.default_rng(seed)
        flip_sets = rng.choice([-1.0, 1.0], size=(n_permutations, n_obs))

    null_maxima = np.empty(n_permutations)
    for index, flips in enumerate(flip_sets):
        permuted = _edgewise_t(samples * flips[:, np.newaxis, np.newaxis])
        _, permuted_masses = _find_clusters(permuted, threshold, signs)
        null_maxima[index] = permuted_masses[0] if permuted_masses.size else 0.0

    # a random null adds the observed labelling to both counts; the exact null
    # already contains it
    extra = 0.0 if exact else 1.0
    cluster_p = np.array(
        [
            (extra + np.sum(null_maxima >= mass)) / (extra + n_permutations)
            for mass in masses
        ]
    )

    significant = np.zeros_like(observed, dtype=bool)
    for mask, p in zip(clusters, cluster_p, strict=True):
        if p <= alpha:
            significant |= mask
    if antisymmetric:
        # both orientations of a pair, as surrogate_test and group_test report
        significant |= significant.T

    return ClusterResult(
        statistic=observed,
        clusters=clusters,
        cluster_p=cluster_p,
        significant=significant,
        threshold=threshold,
        n_permutations=n_permutations,
        alpha=alpha,
    )
