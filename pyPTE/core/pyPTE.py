from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert

BinningRule = Literal["auto", "scott", "hillebrand"] | int
DelayRule = Literal["zero-crossing", "phase-increment"] | int


def get_delay(phase: npt.NDArray[np.float64], rule: DelayRule = "zero-crossing") -> int:
    """
    Estimates the analysis lag, in samples, from the rhythm of the signal

    Both rules estimate the same quantity, half the dominant period, and on a
    narrowband signal they agree to within a sample: for a 10 Hz rhythm at
    250 Hz they return 12 and 13 against a true half-period of 12.5.

    They diverge where the phase itself is poorly defined. On a signal with a
    large DC offset the mean rate of phase advance is badly misled while sign
    counting is not, and on broadband or 1/f signals the two can differ by more
    than a factor of two. Neither is trustworthy there, which is the argument
    for band-pass filtering before estimating anything.

    Note also that a signal containing several rhythms yields a delay tracking
    the fastest of them, not the one of interest.

    Parameters
    ----------
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
    rule : {"zero-crossing", "phase-increment"} or int
        "zero-crossing" counts sign changes across every channel, as in
        Hillebrand et al. 2016, and is the default so results stay comparable.
        "phase-increment" derives the period from how fast the phase advances.
        An integer is taken as the delay itself and returned unchanged.

    Returns
    -------
    delay : int
        at least 1, since a delay of zero would compare a sample with itself

    Raises
    ------
    ValueError
        if the signal has no detectable rhythm, so no lag can be estimated
    """
    if isinstance(rule, (int, np.integer)):
        if rule < 1:
            raise ValueError(f"an explicit delay must be at least 1, got {rule}")
        return int(rule)

    m, n = phase.shape

    if rule == "zero-crossing":
        # compare each sample with its predecessor, excluding the wrap-around
        # pair that np.roll would otherwise introduce at the start of each channel
        crossings = int((phase[:, 1:] * phase[:, :-1] < 0).sum())
        if crossings == 0:
            raise ValueError(
                "no zero crossings found, so the signal has no detectable rhythm; "
                "pass an explicit delay or band-pass filter first"
            )
        return max(int(np.round(n * m / crossings)), 1)

    if rule == "phase-increment":
        # unwrap so the phase advances monotonically, then read off the mean
        # rate; 2*pi over that rate is the period in samples
        advance = np.diff(np.unwrap(phase, axis=1), axis=1)
        mean_advance = float(np.abs(advance).mean())
        if mean_advance <= 0.0:
            raise ValueError(
                "the phase does not advance, so no period can be estimated; "
                "pass an explicit delay or band-pass filter first"
            )
        return max(int(np.round(np.pi / mean_advance)), 1)

    raise ValueError(f"unknown delay rule {rule!r}")


def get_bincount_hillebrand(n_samples: int, delay: int) -> int:
    """
    Bin count as specified by Hillebrand et al. 2016

    Their rule is ``exp(0.626 + 0.4 * ln(n_samples - delay - 1))``, which yields
    three to four times more bins than Scott's rule at the same sample count.
    That leaves the three-way histogram very sparsely populated, so pyPTE uses
    Scott's rule by default; this is provided for comparison with the paper.

    Parameters
    ----------
    n_samples : int
    delay : int

    Returns
    -------
    bincount : int
    """
    usable = n_samples - delay - 1
    if usable < 1:
        raise ValueError(
            f"need more than delay + 1 samples to bin, got {n_samples} with "
            f"delay {delay}"
        )
    return max(int(round(np.exp(0.626 + 0.4 * np.log(usable)))), 2)


def get_bincount_auto(n_samples: int, samples_per_cell: float = 20.0) -> int:
    """
    Bin count sized for the three-dimensional histogram PTE actually fills

    The entropy estimates are biased by roughly the number of occupied cells
    over twice the sample count, and the joint term spreads the samples over
    up to n_bins ** 3 cells. Choosing n_bins = (n_samples / samples_per_cell)
    ** (1/3) keeps the cube populated however long the recording is, where
    Scott's rule, meant for a one-dimensional histogram, leaves about one
    sample per cell and a bias of around 0.75 bits on unrelated signals.

    Parameters
    ----------
    n_samples : int
    samples_per_cell : float
        average samples per cell of the full joint histogram

    Returns
    -------
    bincount : int
        at least 3
    """
    # the small offset keeps exact cubes such as 160000 / 20 = 20 ** 3 from
    # rounding down through floating point
    return max(int(np.floor((n_samples / samples_per_cell) ** (1 / 3) + 1e-9)), 3)


def get_phase(time_series: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Computes phase from time series using a hilbert transform and computing the angles
    between the real and imaginary part for each sample

    Parameters
    ----------
    time_series : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    Returns
    -------
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
    """

    complex_series = hilbert(time_series, axis=1)
    phase: npt.NDArray[np.float64] = np.angle(complex_series)
    return phase


def get_discretized_phase(
    phase: npt.NDArray[np.float64], binsize: float, n_bins: int | None = None
) -> npt.NDArray[np.int32]:
    """
    Discretizes the phase series to rectangular bins

    Parameters
    ----------
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    binsize : float

    n_bins : int, optional
        Number of bins over [0, 2*pi). If not given, inferred from binsize.

    Returns
    -------
    d_phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    """
    if n_bins is None:
        n_bins = get_bincount(binsize)
    d_phase = np.clip(np.floor(phase / binsize), 0, n_bins - 1).astype(np.int32)
    return d_phase


def get_binsize(phase: npt.NDArray[np.float64], c: float = 3.49) -> float:
    """
    Computes the bin size for the phase binning

    Parameters
    ----------
    c : float
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    Returns
    -------
    bincount : float

    """

    m, n = phase.shape
    binsize = c * np.mean(np.std(phase, axis=1, ddof=1)) * n ** (-1.0 / 3)
    return float(binsize)


def get_bincount(binsize: float) -> int:
    """
    Get bin count for the interval [0, 2*pi] for given binsize

    Parameters
    ----------
    binsize : float

    Returns
    -------
    bincount : int

    """
    bins_w = np.arange(0, 2 * np.pi, binsize)
    bincount = len(bins_w)
    return bincount


def _entropy(
    codes: npt.NDArray[np.integer], n_samples: int, bias_correction: bool = False
) -> float:
    """
    Shannon entropy in bits of a discrete variable given its integer states

    Counting pre-encoded joint states with numpy.bincount avoids materialising
    a dense histogram over the full product space, which for the three-way term
    grows with the cube of the number of phase bins while staying almost
    entirely empty.

    Parameters
    ----------
    codes : numpy.ndarray
        1-d array of non-negative integers, one state per sample
    n_samples : int
        number of samples the states were drawn from
    bias_correction : bool
        add the Miller-Madow term (occupied states - 1) / (2 n ln 2), which
        removes the first-order underestimate of the plug-in estimator

    Returns
    -------
    entropy : float

    """
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    # -sum(p*log2(p)) with p = c/n, rearranged so no division happens per bin
    # and empty bins are dropped before the logarithm rather than after
    entropy = np.log2(n_samples) - (counts * np.log2(counts)).sum() / n_samples
    if bias_correction:
        entropy += (counts.size - 1) / (2 * n_samples * np.log(2))
    return float(entropy)


def compute_PTE(
    phase: npt.NDArray[np.integer],
    delay: int,
    n_bins: int | None = None,
    bias_correction: bool = False,
) -> npt.NDArray[np.float64]:
    """
    For each channel pair (x, y) containing the individual discretized phase,
    which is obtained by pyPTE.pyPTE.get_discretized_phase,
    this function performs the entropy estimation by counting the occurences of
    phase values in x, y and y_predicted, which is achieved by slicing the x, y
    to consider delay x samples in the past and delay samples in the future.

    Joint states are packed into a single integer per sample, treating the
    phase bin of each variable as one digit in base n_bins, so every entropy
    reduces to a one-dimensional bincount.

    Parameters
    ----------
    phase : numpy.ndarray
         m x n ndarray : m: number of channels, n: number of samples
    delay : int
        This is the analysis delta, which is the number of samples in the past
        to be considered for x and y. Momentarily delay is estimated by
        pyPTE.pyPTE.get_delay(). A custom delay estimation can be used as well.
    n_bins : int, optional
        Number of bins over [0, 2*pi). If not given, inferred as int(phase.max()) + 1.
    bias_correction : bool
        apply the Miller-Madow correction to every entropy term, see _entropy

    Returns
    -------
    PTE : numpy.ndarray
        m x m matrix containing the PTE value for each channel pair

    Raises
    ------
    ValueError
        if a phase code falls outside 0 .. n_bins - 1, where joint states
        would collide silently, or if delay leaves no samples to count
    """
    m, n = phase.shape
    n_samples = n - delay
    if n_samples < 1:
        raise ValueError(
            f"delay must be shorter than the recording, got {delay} for {n} samples"
        )
    if n_bins is None:
        n_bins = int(phase.max()) + 1
    if phase.min() < 0 or phase.max() >= n_bins:
        raise ValueError(
            f"phase codes must lie in 0 .. {n_bins - 1}, got "
            f"{int(phase.min())} .. {int(phase.max())}"
        )

    # int64 keeps the three-digit code below the overflow limit for the bin
    # counts Scott's rule produces on realistically long recordings
    ypr_all = phase[:, delay:].astype(np.int64)
    y_all = phase[:, :-delay].astype(np.int64)

    PTE = np.zeros((m, m), dtype=float)

    # the target channel drives the outer loop because H(y) and H(ypr, y)
    # depend on it alone, so they are computed m times rather than m * m
    for j in range(m):
        y = y_all[j]
        ypr = ypr_all[j]

        Hy = _entropy(y, n_samples, bias_correction)
        Hypr_y = _entropy(ypr * n_bins + y, n_samples, bias_correction)

        # the source occupies the least significant digit, so everything above
        # it can be shifted once here and reused for every source channel
        y_shifted = y * n_bins
        ypr_y_shifted = (ypr * n_bins + y) * n_bins

        for i in range(m):
            if i == j:
                # self-transfer cancels exactly, since H(y, y) == H(y) and
                # H(ypr, y, y) == H(ypr, y)
                continue

            x = y_all[i]
            Hy_x = _entropy(y_shifted + x, n_samples, bias_correction)
            Hypr_y_x = _entropy(ypr_y_shifted + x, n_samples, bias_correction)

            PTE[i, j] = Hypr_y + Hy_x - Hy - Hypr_y_x

    if not bias_correction:
        # the plug-in estimate is never negative; summing four entropies leaves
        # rounding noise around zero, e.g. for identical channels, which would
        # otherwise turn 0/0 in dPTE into an arbitrary number
        PTE[np.abs(PTE) < 1e-12] = 0.0
    return PTE


def compute_dPTE_rawPTE(
    phase: npt.NDArray[np.integer],
    delay: int,
    n_bins: int | None = None,
    bias_correction: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    This function calls pyPTE.pyPTE.compute_PTE to obtain a PTE matrix and performs a
    normalization yielding dPTE to easily investigate directionality information.
    Technically it could be a function which computes the normalization for a given
    PTE matrix, but it appears to be more convenient to obtain both matrices in one call

    Parameters
    ----------
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
        The discretized phase is computed by pyPTE.pyPTE.get_discretized_phase

    delay : int
        This is the analysis delta, which is the number of samples in the past to be
        considered for x and y. Momentarily delay is estimated by
        pyPTE.pyPTE.get_delay(). A custom delay estimation can be used as well.

    n_bins : int, optional
        Number of bins over [0, 2*pi). Passed to compute_PTE.

    bias_correction : bool
        passed to compute_PTE. Corrected values can dip below zero, so dPTE is
        formed from them clipped at zero.

    Returns
    -------
    (dPTE, raw_PTE) : tuple of numpy.ndarray objects
        dPTE : normalized PTE matrix, raw_PTE: original PTE values. dPTE is NaN
        for pairs with no transfer in either direction, where it is 0/0.

    """
    raw_PTE = compute_PTE(phase, delay, n_bins=n_bins, bias_correction=bias_correction)

    transfer = np.clip(raw_PTE, 0.0, None)
    total = transfer + transfer.T
    with np.errstate(divide="ignore", invalid="ignore"):
        dPTE = np.where(total > 0, transfer / total, np.nan)
    np.fill_diagonal(dPTE, 0.0)
    return dPTE, raw_PTE


def PTE(
    time_series: npt.ArrayLike,
    *,
    binning: BinningRule = "scott",
    delay: DelayRule = "zero-crossing",
    bias_correction: bool = False,
    trim_cycles: float = 0.0,
    time_reversal: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes dPTE and raw PTE for every ordered channel pair

    The procedure is:

    1. Compute the phase with a Hilbert transform, on the interval [-pi, pi[
    2. Estimate the analysis delay, and optionally trim the edges
    3. Shift the phase so there are no negative values, then bin it
    4. Count joint states and combine the entropies

    Band-pass the input narrowly to the rhythm of interest first. On broadband
    data dPTE points from the cleaner channel to the noisier one, whatever the
    coupling, and even after filtering a large enough difference in
    signal-to-noise ratio can reverse it. ``time_reversal=True`` cancels that
    artefact; see the analysis guide.

    Both free choices in that procedure are exposed, because neither is
    obviously correct and they change the answer. Wider bins give a
    better-populated histogram and a larger apparent effect; the delay sets
    which lag the measure is sensitive to.

    Parameters
    ----------
    time_series : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
    binning : {"scott", "auto", "hillebrand"} or int
        "scott", the default, sizes bins by Scott's rule. The fine bins keep
        the phase resolution dPTE depends on, and the estimator's bias cancels
        in the dPTE ratio. Raw PTE under it is biased upwards, by about 0.75
        bits on unrelated signals, so do not read raw values on their own.
        "auto" sizes bins for the three-dimensional joint histogram (see
        get_bincount_auto), which keeps raw PTE close to unbiased but blurs
        phase and usually weakens dPTE. Use it when raw PTE in bits matters.
        "hillebrand" uses the bin count published in Hillebrand et al. 2016,
        which is finer still and, in simulation, more easily turned around by
        a difference in signal-to-noise ratio. An integer sets the number of
        bins directly.
    delay : {"zero-crossing", "phase-increment"} or int
        passed to pyPTE.core.pyPTE.get_delay; an integer is used as-is
    bias_correction : bool
        apply the Miller-Madow correction, which brings raw PTE on unrelated
        signals to about zero. Use it when the size of raw PTE in bits matters.
        It makes dPTE noisier, so leave it off when only direction matters.
    trim_cycles : float
        dominant cycles to drop from each end after the Hilbert transform,
        where its phase is least reliable. One cycle is twice the estimated
        delay. 0, the default, keeps every sample; 1 to 3 is typical for long
        recordings, but short epochs may not have samples to spare.
    time_reversal : bool
        return the time-reversal-differenced dPTE instead of dPTE (see
        reversal_differenced_dPTE). It cancels directions that come from a
        difference in signal-to-noise ratio or from instantaneous mixing,
        which plain dPTE reports as real, at twice the cost. raw_PTE is still
        the forward raw PTE.

    Returns
    -------
    (dPTE, raw_PTE) : tuple of numpy.ndarray objects
        dPTE : normalized PTE matrix, raw_PTE : PTE in bits. Entry [i, j]
        describes information flow from channel i to channel j.

    """
    d_phase, analysis_delay, n_bins = prepare_phase(
        time_series, binning=binning, delay=delay, trim_cycles=trim_cycles
    )
    dPTE, raw_PTE = compute_dPTE_rawPTE(
        d_phase, analysis_delay, n_bins=n_bins, bias_correction=bias_correction
    )
    if time_reversal:
        reversed_dPTE, _ = compute_dPTE_rawPTE(
            reverse_phase(d_phase),
            analysis_delay,
            n_bins=n_bins,
            bias_correction=bias_correction,
        )
        dPTE = reversal_differenced_dPTE(dPTE, reversed_dPTE)
    return dPTE, raw_PTE


def PTE_segments(
    time_series: npt.ArrayLike,
    n_segments: int,
    *,
    binning: BinningRule = "scott",
    delay: DelayRule = "zero-crossing",
    bias_correction: bool = False,
    trim_cycles: float = 0.0,
    time_reversal: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes one dPTE and raw PTE matrix per segment of a continuous recording

    The way to put a group test on one long recording: the matrices go to
    pyPTE.group_test, which asks whether a direction is consistent across
    segments. With ``time_reversal=True`` that is the robust test of
    direction, since directions produced by a difference in signal-to-noise
    ratio or by instantaneous mixing cancel in every segment. The surrogate
    test cannot do this for the differenced statistic, because its null
    assumes independent channels.

    The phase is computed once over the whole recording, so the segments add
    no Hilbert edges, and every segment shares one delay and one bin count.

    Parameters
    ----------
    time_series : numpy.ndarray
        m x n array : m channels, n samples
    n_segments : int
        number of contiguous, near-equal segments. With Wilcoxon's test the
        smallest two-sided p-value over n segments is 2 / 2**n, so the count
        bounds what can survive correction; group_test warns when it is too
        small. Each segment should still span many cycles of the rhythm.
    binning, delay, bias_correction, trim_cycles, time_reversal
        as for `PTE`

    Returns
    -------
    (dPTE, raw_PTE) : tuple of numpy.ndarray objects
        n_segments x m x m arrays, ready for pyPTE.group_test. raw_PTE is the
        forward raw PTE even with time_reversal.
    """
    if n_segments < 2:
        raise ValueError(f"n_segments must be at least 2, got {n_segments}")
    d_phase, analysis_delay, n_bins = prepare_phase(
        time_series, binning=binning, delay=delay, trim_cycles=trim_cycles
    )
    if d_phase.shape[1] // n_segments <= analysis_delay:
        raise ValueError(
            f"{n_segments} segments of {d_phase.shape[1] // n_segments} samples "
            f"are too short for the delay of {analysis_delay} samples"
        )
    return compute_segments(
        np.array_split(d_phase, n_segments, axis=1),
        analysis_delay,
        n_bins,
        bias_correction=bias_correction,
        time_reversal=time_reversal,
    )


def compute_segments(
    segments: Sequence[npt.NDArray[np.int32]],
    delay: int,
    n_bins: int,
    *,
    bias_correction: bool = False,
    time_reversal: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    dPTE and raw PTE for each of several binned phase arrays

    Parameters
    ----------
    segments : sequence of numpy.ndarray
        m x n_k binned phase arrays, as from `prepare_phase`, sharing delay and
        bins; lengths may differ
    delay, n_bins
        the shared delay and bin count
    bias_correction, time_reversal
        as for `PTE`

    Returns
    -------
    (dPTE, raw_PTE) : tuple of numpy.ndarray objects
        len(segments) x m x m arrays
    """
    m = segments[0].shape[0]
    dPTE = np.empty((len(segments), m, m))
    raw_PTE = np.empty((len(segments), m, m))
    for k, segment in enumerate(segments):
        segment = np.ascontiguousarray(segment)
        dPTE[k], raw_PTE[k] = compute_dPTE_rawPTE(
            segment, delay, n_bins=n_bins, bias_correction=bias_correction
        )
        if time_reversal:
            reversed_dPTE, _ = compute_dPTE_rawPTE(
                reverse_phase(segment),
                delay,
                n_bins=n_bins,
                bias_correction=bias_correction,
            )
            dPTE[k] = reversal_differenced_dPTE(dPTE[k], reversed_dPTE)
    return dPTE, raw_PTE


def reverse_phase(d_phase: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """The binned phase with time running backwards.

    Reversing the binned phase, rather than binning the reversed signal, makes
    the reverse of a time-shifted surrogate exactly the surrogate of the
    reverse, which the differenced test relies on.
    """
    return np.ascontiguousarray(d_phase[:, ::-1])


def reversal_differenced_dPTE(
    dPTE: npt.NDArray[np.float64], reversed_dPTE: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Combines dPTE of a recording and of its time reverse into one direction

    A real interaction reverses with time, so it pushes dPTE and the dPTE of
    the reversed recording to opposite sides of 0.5. A difference in
    signal-to-noise ratio, or one source mixed instantaneously into two
    channels, looks the same either way round, and pushes both to the same
    side. Taking half the difference keeps the first and cancels the second:
    the time-reversal-differenced statistic of Haufe et al. (2012) and
    Winkler et al. (2016), applied to dPTE.

    The cancellation is shown in simulation, not proven: Winkler et al. prove
    it for linear models, and nonlinear coupling can survive time reversal
    (Kořenek & Hlinka 2021). A common driver that reaches two channels at
    different lags still reads as a direction, since that order is real.

    Parameters
    ----------
    dPTE, reversed_dPTE : numpy.ndarray
        m x m dPTE of the recording and of its time reverse, computed with the
        same delay and bins

    Returns
    -------
    differenced : numpy.ndarray
        m x m matrix on the dPTE scale, ``0.5 + (dPTE - reversed_dPTE) / 2``.
        Like dPTE it is antisymmetric about 0.5, so it works with group_test
        and the other group statistics. The diagonal is zero.
    """
    differenced = 0.5 + (dPTE - reversed_dPTE) / 2
    np.fill_diagonal(differenced, 0.0)
    return differenced


def prepare_phase(
    time_series: npt.ArrayLike,
    *,
    binning: BinningRule = "scott",
    delay: DelayRule = "zero-crossing",
    trim_cycles: float = 0.0,
) -> tuple[npt.NDArray[np.int32], int, int]:
    """
    Turns raw signals into the binned phase that PTE counts

    Steps 1 to 3 of PTE, exposed so that a caller can fix the delay and bins
    once and reuse them, as surrogate_test does for every surrogate.

    Parameters
    ----------
    time_series : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
    binning, delay, trim_cycles
        as for PTE

    Returns
    -------
    (d_phase, delay, n_bins) : tuple
        m x n' array of phase bins in 0 .. n_bins - 1, where n' is n less any
        trimmed samples; the analysis delay in samples; the number of bins

    """
    if trim_cycles < 0:
        raise ValueError(f"trim_cycles must be at least 0, got {trim_cycles}")

    phase = get_phase(time_series)
    analysis_delay = get_delay(phase, delay)

    if trim_cycles > 0:
        # the delay estimates half the dominant period, so one cycle is twice it
        trim = int(round(trim_cycles * 2 * analysis_delay))
        if phase.shape[1] - 2 * trim <= analysis_delay:
            raise ValueError(
                f"trimming {trim_cycles} cycles ({trim} samples at each end) "
                f"leaves too little of a {phase.shape[1]}-sample recording"
            )
        if trim > 0:
            phase = phase[:, trim:-trim]

    phase_inc = phase + np.pi
    n_samples = phase_inc.shape[1]

    if binning == "auto":
        n_bins = get_bincount_auto(n_samples - analysis_delay)
        binsize = 2 * np.pi / n_bins
    elif isinstance(binning, (int, np.integer)):
        if binning < 2:
            raise ValueError(f"binning needs at least 2 bins, got {binning}")
        n_bins = int(binning)
        binsize = 2 * np.pi / n_bins
    elif binning == "scott":
        binsize = get_binsize(phase_inc)
        n_bins = get_bincount(binsize)
    elif binning == "hillebrand":
        n_bins = get_bincount_hillebrand(n_samples, analysis_delay)
        binsize = 2 * np.pi / n_bins
    else:
        raise ValueError(f"unknown binning rule {binning!r}")

    d_phase = get_discretized_phase(phase_inc, binsize, n_bins=n_bins)
    return d_phase, analysis_delay, n_bins
