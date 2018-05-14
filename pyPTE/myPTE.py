import numpy as np
from scipy.signal import hilbert


def _get_delay(phase):
    phase = phase
    m, n = phase.shape
    c1 = n*(m-2)
    r_phase = np.roll(phase, 2, axis=0)
    m = np.multiply(phase, r_phase)[1:-1]
    c2 = (m < 0).sum()
    delay = int(np.round(c1/c2))
    return delay


def _get_phase(time_series):
    """
    Computes phase from time series using a hilbert transform and computing the angles between the real and imaginary part for each sample

    Parameters
    ----------
    time_series : ndarray
        m x n ndarray : m: number of channels, n: number of samples

    Returns
    -------
    phase : ndarray
        m x n ndarray : m: number of channels, n: number of samples
    """

    complex_series = hilbert(time_series, axis=0)
    phase = np.angle(complex_series)
    return phase

def _get_discretized_phase(phase, binsize):
    """
    Discretizes the phase series to rectangular bins

    Parameters
    ----------
    phase : ndarray
        m x n ndarray : m: number of channels, n: number of samples

    binsize : float

    Returns
    -------
    d_phase : ndarray
        m x n ndarray : m: number of channels, n: number of samples

    """
    d_phase = np.ceil(phase / binsize).astype(np.int32)
    return d_phase


def _get_binsize(phase, c = 3.49):

    """
    Computes the bin size for the phase binning

    Parameters
    ----------
    c : float
    phase : ndarray
        m x n ndarray : m: number of channels, n: number of samples
    method

    Returns
    -------
    bincount : float

    """

    m, n = phase.shape
    binsize = c * np.mean(np.std(phase, axis=0, ddof=1)) * m ** (-1.0 / 3)
    return binsize

def _get_bincount(binsize):
    """
    Get bin count for the interval [0, 2*pi] for giving binsize

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


def _compute_PTE(phase, delay):

    m, n = phase.shape
    PTE = np.zeros((n,n), dtype=float)

    sc1 = (m-delay)
    sc2 = np.log2(sc1)
    for i in range(0, n):
        for j in range(0, n):
    # it = np.nditer(PTE, flags=['multi_index'], op_flags=['writeonly'])
    # while not it.finished:
    #     i = it.multi_index[0]
    #     j = it.multi_index[1]

            ypr = phase[delay:, j]
            y = phase[:-delay, j]
            x = phase[:-delay, i]

            P_y = np.zeros([y.max() +1])
            np.add.at(P_y, [y], 1)

            P_ypr_y = np.zeros([ypr.max()+1, y.max()+1])
            np.add.at(P_ypr_y, [ypr, y], 1)

            P_y_x = np.zeros([y.max()+1, x.max()+1])
            np.add.at(P_y_x, [y, x], 1)

            P_ypr_y_x = np.zeros([ypr.max()+1, y.max()+1, x.max()+1])
            np.add.at(P_ypr_y_x, [ypr, y, x], 1)

            P_y /= (m-delay)
            P_ypr_y /= (m-delay)
            P_y_x /= (m-delay)
            P_ypr_y_x /= (m-delay)

            Hy = -np.nansum(np.multiply(P_y,np.log2(P_y)))
            Hypr_y = - np.nansum(np.nansum(np.multiply(P_ypr_y, np.log2(P_ypr_y))))
            Hy_x = -np.nansum(np.nansum(np.multiply(P_y_x, np.log2(P_y_x))))
            Hypr_y_x = -np.nansum(np.nansum(np.nansum(np.multiply(P_ypr_y_x, np.log2(P_ypr_y_x)))))
            PTE[i, j] = Hypr_y + Hy_x - Hy - Hypr_y_x
    # it.iternext()
    return PTE

def compute_dPTE_rawPTE(phase, delay):
    raw_PTE = _compute_PTE(phase, delay)

    tmp = np.triu(raw_PTE) + np.tril(raw_PTE).T
    with np.errstate(divide='ignore',invalid='ignore'):
        dPTE = np.triu(raw_PTE/tmp,1) + np.tril(raw_PTE/tmp.T,-1)
    return dPTE, raw_PTE
