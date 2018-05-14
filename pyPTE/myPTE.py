import numpy as np
from scipy.signal import hilbert


# def _get_delay(phase):
#     """
#     Calculates the the overall delay for all phase-channel-pairs in a measurement
#
#     WARNING: only applicable if all pairwise delays are expected to be more or less the same
#
#     Parameters
#     ----------
#     phase : ndarray
#         m x n ndarray : m: number of channels, n: number of samples
#
#     Returns
#     -------
#     delay : int
#         number of time steps, average number of pairwise different signs
#
#     """
#     m, n = phase.shape
#     counter1 = 0
#     counter2 = 0
#     for j in range(0, n):
#         for i in range(1, m-1):
#             counter1 += 1
#             if (((phase[i-1, j])*(phase[i+1, j] )) < 0):
#                 counter2 += 1
#     delay = int(np.round(float(counter1)/float(counter2)))
#     return delay

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



# def _compute_PTE(phase, delay, Nbins):
#
#     m, n = phase.shape
#
#     # print(delay, m, n)
#     PTE = np.zeros((n,n), dtype=float)
#
#     sc1 = (m-delay)
#     sc2 = np.log2(sc1)
#
#     for i in range(0, n):
#         for j in range(0, n):
#
#             if j != i:
#                 Py = np.empty((Nbins, 1), dtype=float)
#                 Pypr_y = np.empty((Nbins, Nbins), dtype=float)
#                 Py_x = np.empty((Nbins, Nbins), dtype=float)
#                 Pypr_y_x = np.empty((Nbins, Nbins, Nbins), dtype=float)
#
#                 # rn_ypr = (np.ceil(phase[delay:, j]/binsize))
#                 # rn_y = (np.ceil(phase[:-delay, j]/binsize))
#                 # rn_x = (np.ceil(phase[:-delay, i]/binsize))
#
#
#                 for k in range(0, m-delay):
#                     # rn_ypr_k = phase[delay+k, j] -1
#                     # rn_y_k = phase[k, j] - 1
#                     # rn_x_k = phase[k, i] - 1
#
#                     rn_ypr_k = phase[delay+k, j]
#                     rn_y_k = phase[k, j]
#                     rn_x_k = phase[k, i]
#
#                     Py[rn_y_k] += 1
#                     Pypr_y[rn_ypr_k, rn_y_k] += 1
#                     Py_x[rn_y_k, rn_x_k] += 1
#                     Pypr_y_x[rn_ypr_k, rn_y_k, rn_x_k] += 1
#
#                     # Py[rn_y[k]] += 1
#                     # Pypr_y[rn_ypr[k], rn_y[k]] += 1
#                     # Py_x[rn_y[k], rn_x[k]] += 1
#                     # Pypr_y_x[rn_ypr[k], rn_y[k], rn_x[k]] += 1
#
#                 Py /= (m-delay)
#                 Pypr_y /= (m-delay)
#                 Py_x /= (m-delay)
#                 Pypr_y_x /= (m-delay)
#                 # print(Py, Pypr_y, Py_x, Pypr_y_x)
#
#                 Hy = -np.nansum(np.multiply(Py,np.log2(Py)))
#                 Hypr_y = - np.nansum(np.nansum(np.multiply(Pypr_y, np.log2(Pypr_y))))
#                 Hy_x = -np.nansum(np.nansum(np.multiply(Py_x, np.log2(Py_x))))
#                 Hypr_y_x = -np.nansum(np.nansum(np.nansum(np.multiply(Pypr_y_x, np.log2(Pypr_y_x)))))
#                 PTE[i, j] = Hypr_y + Hy_x - Hy - Hypr_y_x
#
#
#                 # Hy = 0
#                 # Hypr_y = 0
#                 # Hy_x = 0
#                 # Hypr_y_x = 0
#                 #
#                 # for ii in range(0, Nbins-1):
#                 #     if (Py[ii] > 0):
#                 #         Hy += Py[ii] * (np.log2(Py[ii]) - sc2)
#                 #     for jj in range(0 , Nbins-1):
#                 #         v = Pypr_y[ii, jj]
#                 #         if (v > 0):
#                 #             Hypr_y += v * (np.log2(v) - sc2)
#                 #         if (v > 0):
#                 #             Hy_x += v * (np.log2(v) - sc2)
#                 #         for kk in range(0, Nbins-1):
#                 #             v = Pypr_y_x[ii, jj, kk]
#                 #         if (v > 0):
#                 #             Hypr_y_x += v * (np.log2(v)-sc2)
#                 # print((Hypr_y + Hy_x - Hy - Hypr_y_x)/sc1)
#                 # PTE[i, j] = (Hypr_y + Hy_x - Hy - Hypr_y_x)/sc1
#
#
#     return PTE


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
