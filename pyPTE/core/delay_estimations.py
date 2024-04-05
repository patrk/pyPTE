import numpy as np
import numpy.typing as npt


def delay_by_hillebrand(phase: npt.NDArray) -> npt.NDArray:
    """
    Original method to compute the overall delay for all given channels.

    Parameters:
    ----------
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples.

    Returns:
    -------
    delay : int
        The computed delay.
    """
    m, n = phase.shape
    c1 = n * m
    r_phase = np.roll(phase, 1, axis=0)
    phase_product = np.multiply(phase, r_phase)
    c2 = (phase_product < 0).sum()
    delay = int(np.round(c1 / c2))
    delay_matrix = np.full((m, m), delay, dtype=int)
    return delay_matrix

def delay_by_crosscorrelation(phase_matrix: npt.NDArray) -> npt.NDArray:
    m, _ = phase_matrix.shape
    delay_matrix = np.zeros((m, m), dtype=int)

    for i in range(m):
        for j in range(i+1, m):
            unwrapped_phase_i = np.unwrap(phase_matrix[i])
            unwrapped_phase_j = np.unwrap(phase_matrix[j])

            cross_corr = np.correlate(unwrapped_phase_i - np.mean(unwrapped_phase_i),
                                   unwrapped_phase_j - np.mean(unwrapped_phase_j),
                                   mode='full')

            delay = np.argmax(cross_corr) - (len(unwrapped_phase_i) - 1)
            delay_matrix[i, j] = delay

    delay_matrix = delay_matrix - delay_matrix.T

    return delay_matrix

