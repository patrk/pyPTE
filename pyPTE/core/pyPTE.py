from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert


def get_delay(phase: npt.NDArray) -> int:
    """
    Computes the overall delay for a all given channels

    Parameters
    ----------
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    Returns
    -------
    delay : int
    """
    phase = phase
    m, n = phase.shape
    c1 = n * m
    r_phase = np.roll(phase, 1, axis=1)
    phase_product = np.multiply(phase, r_phase)
    c2 = (phase_product < 0).sum()
    delay = int(np.round(c1 / c2))

    return delay


def get_phase(time_series: npt.ArrayLike) -> npt.NDArray:
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
    phase = np.angle(complex_series)
    return phase


def get_discretized_phase(phase: npt.NDArray, binsize: float) -> npt.NDArray:
    """
    Discretizes the phase series to rectangular bins

    Parameters
    ----------
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    binsize : float

    Returns
    -------
    d_phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    """
    d_phase = np.ceil(phase / binsize).astype(np.int32)
    return d_phase


def get_binsize(phase: npt.NDArray, c: float = 3.49) -> float:
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
    return binsize


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


def _entropy(codes: npt.NDArray, n_samples: int) -> float:
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

    Returns
    -------
    entropy : float

    """
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    # -sum(p*log2(p)) with p = c/n, rearranged so no division happens per bin
    # and empty bins are dropped before the logarithm rather than after
    return float(np.log2(n_samples) - (counts * np.log2(counts)).sum() / n_samples)


def compute_PTE(phase: npt.NDArray, delay: int) -> npt.NDArray:
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

    Returns
    -------
    PTE : numpy.ndarray
        m x m matrix containing the PTE value for each channel pair
    """
    m, n = phase.shape
    n_samples = n - delay
    n_bins = int(phase.max()) + 1

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

        Hy = _entropy(y, n_samples)
        Hypr_y = _entropy(ypr * n_bins + y, n_samples)

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
            Hy_x = _entropy(y_shifted + x, n_samples)
            Hypr_y_x = _entropy(ypr_y_shifted + x, n_samples)

            PTE[i, j] = Hypr_y + Hy_x - Hy - Hypr_y_x

    return PTE


def compute_dPTE_rawPTE(
    phase: npt.NDArray, delay: int
) -> Tuple[npt.NDArray, npt.NDArray]:
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

    Returns
    -------
    (dPTE, raw_PTE) : tuple of numpy.ndarray objects
        dPTE : normalized PTE matrix, raw_PTE: original PTE values

    """
    raw_PTE = compute_PTE(phase, delay)

    tmp = np.triu(raw_PTE) + np.tril(raw_PTE).T
    with np.errstate(divide="ignore", invalid="ignore"):
        dPTE = np.triu(raw_PTE / tmp, 1) + np.tril(raw_PTE / tmp.T, -1)
    return dPTE, raw_PTE


def PTE(time_series: npt.ArrayLike) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    This function performs the whole procedure of calculating the PTE:
    1. Compute the phase by applying the Hilbert transform on the time-series and
    calculate the angle between the real and imaginary part.
    The phase is defined on the interval [-pi, pi[
    2. Estimate the analysis delay
    3. For binning, shift the phase along the ordinate so there are no negatives values
    4. Calculate the binsize in number of samples
    5. Bin the phase data
    6. Compute the dPTE and raw_PTE

    Parameters
    ----------
    time_series : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    Returns
    -------
    (dPTE, raw_PTE) : tuple of numpy.ndarray objects
        dPTE : normalized PTE matrix, raw_PTE: original PTE values

    """
    phase = get_phase(time_series)
    delay = get_delay(phase)
    phase_inc = phase + np.pi
    binsize = get_binsize(phase_inc)
    d_phase = get_discretized_phase(phase_inc, binsize)

    return compute_dPTE_rawPTE(d_phase, delay)
