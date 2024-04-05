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
    r_phase = np.roll(phase, 1, axis=0)
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

    complex_series = hilbert(time_series, axis=0)
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
    binsize = c * np.mean(np.std(phase, axis=0, ddof=1)) * n ** (-1.0 / 3)
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


def compute_PTE(phase: npt.NDArray, delay: int) -> npt.NDArray:
    """
    For each channel pair (x, y) containing the individual discretized phase,
    which is obtained by pyPTE.pyPTE.get_discretized_phase,
    this function performs the entropy estimation by counting the occurences of
    phase values in x, y and y_predicted, which is achieved by slicing the x, y
    to consider delay x samples in the past and delay samples in the future.

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
    PTE = np.zeros((m, m), dtype=float)

    for i in range(0, m):
        for j in range(0, m):

            ypr = phase[delay:, j]
            y = phase[:-delay, j]
            x = phase[:-delay, i]

            P_y = np.zeros([y.max() + 1])
            np.add.at(P_y, [y], 1)

            max_dim_ypr_y = max(ypr.max(), y.max()) + 1
            P_ypr_y = np.zeros([max_dim_ypr_y, max_dim_ypr_y])

            max_dim_y_x = max(y.max(), x.max()) + 1
            P_y_x = np.zeros([max_dim_y_x, max_dim_y_x])

            max_dim_ypr_y_x = max(ypr.max(), y.max(), x.max()) + 1
            P_ypr_y_x = np.zeros([max_dim_ypr_y_x, max_dim_ypr_y_x, max_dim_ypr_y_x])

            P_y /= m - delay
            P_ypr_y /= m - delay
            P_y_x /= m - delay
            P_ypr_y_x /= m - delay

            Hy = -np.nansum(np.multiply(P_y, np.log2(P_y)))
            Hypr_y = -np.nansum(np.nansum(np.multiply(P_ypr_y, np.log2(P_ypr_y))))
            Hy_x = -np.nansum(np.nansum(np.multiply(P_y_x, np.log2(P_y_x))))
            Hypr_y_x = -np.nansum(
                np.nansum(np.nansum(np.multiply(P_ypr_y_x, np.log2(P_ypr_y_x))))
            )
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
