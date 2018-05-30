import numpy as np
import pandas as pd
from scipy.signal import hilbert


def get_delay(phase):
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
    c1 = n*(m-2)
    r_phase = np.roll(phase, 2, axis=0)
    m = np.multiply(phase, r_phase)[1:-1]
    c2 = (m < 0).sum()
    delay = int(np.round(c1/c2))
    return delay


def get_phase(time_series):
    """
    Computes phase from time series using a hilbert transform and computing the angles between the real and imaginary part for each sample

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

def get_discretized_phase(phase, binsize):
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


def get_binsize(phase, c = 3.49):

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
    binsize = c * np.mean(np.std(phase, axis=0, ddof=1)) * m ** (-1.0 / 3)
    return binsize

def get_bincount(binsize):
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


def compute_PTE(phase, delay):
    """
    For each channel pair (x, y) containing the individual discretized phase, which is obtained by pyPTE.pyPTE.get_discretized_phase,
    this function performs the entropy estimation by counting the occurences of phase values in x, y and y_predicted,
    which is achieved by slicing the x, y to consider delay x samples in the past and delay samples in the future.
    Parameters
    ----------
    phase : numpy.ndarray
         m x n ndarray : m: number of channels, n: number of samples
    delay : int
        This is the analysis delta, which is the number of samples in the past to be considered for x and y
        Momentarily delay is estimated by pyPTE.pyPTE.get_delay(). A custom delay estimation can be used as well.

    Returns
    -------
    PTE : numpy.ndarray
        m x m matrix containing the PTE value for each channel pair
    """
    m, n = phase.shape
    PTE = np.zeros((n,n), dtype=float)

    for i in range(0, n):
        for j in range(0, n):

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
    return PTE

def compute_dPTE_rawPTE(phase, delay):
    """

    Parameters
    ----------
    phase
    delay

    Returns
    -------

    """
    raw_PTE = compute_PTE(phase, delay)

    tmp = np.triu(raw_PTE) + np.tril(raw_PTE).T
    with np.errstate(divide='ignore',invalid='ignore'):
        dPTE = np.triu(raw_PTE/tmp,1) + np.tril(raw_PTE/tmp.T,-1)
    return dPTE, raw_PTE

def PTE(time_series):

    phase = get_phase(time_series)
    delay = get_delay(phase)
    phase_inc = phase + np.pi
    binsize = get_binsize(phase_inc)
    d_phase = get_discretized_phase(phase_inc, binsize)

    return compute_dPTE_rawPTE(d_phase, delay)

def PTE_from_dataframe(data_frame):
    time_series = data_frame.as_matrix()
    dPTE, rPTE = PTE(time_series)
    dPTE_df = pd.DataFrame(dPTE, index=data_frame.columns, columns=data_frame.columns)
    rPTE_df = pd.DataFrame(rPTE, index=data_frame.columns, columns=data_frame.columns)
    return dPTE_df, rPTE_df




