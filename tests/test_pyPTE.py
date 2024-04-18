import numpy as np

from pyPTE.core.pyPTE import (
    PTE,
    compute_PTE,
    get_binsize,
    get_delay,
    get_discretized_phase,
    get_phase,
)


def test_get_phase_zeros_input():
    time_series = np.array([[0, 0, 0], [0, 0, 0]])
    expected_phase_result = np.array([[0, 0, 0], [0, 0, 0]])
    np.testing.assert_almost_equal(
        get_phase(time_series),
        expected_phase_result, decimal=5)

def test_get_discretized_phase():
    phase_data = np.array([np.pi/4, np.pi/2, 3*np.pi/4])
    binsize = np.pi/4
    expected_discretized_phase = np.array([1, 2, 3])
    discretized_phase = get_discretized_phase(phase_data, binsize)
    np.testing.assert_array_equal(
        discretized_phase,
        expected_discretized_phase,
        "Discretized phase did not match expected values.")

def test_function_shapes():
    time_series = np.random.rand(4, 150)  # 4 channels, 150 samples

    phase = get_phase(time_series)
    assert phase.shape == (4, 150), "get_phase output shape mismatch"

    delay = get_delay(phase)
    assert isinstance(delay, int), "get_delay output should be an integer"

    binsize = get_binsize(phase)
    assert isinstance(binsize, float), "get_binsize output should be a float"

    d_phase = get_discretized_phase(phase, binsize)
    assert d_phase.shape == (4, 150), "get_discretized_phase output shape mismatch"

    dPTE, raw_PTE = PTE(time_series)
    assert dPTE.shape == (4, 4),  (
        f"Expected rawPTE shape (4, 4), got {dPTE.shape}")
    assert raw_PTE.shape == (4, 4),  (
        f"Expected dPTE shape (4, 4), got {dPTE.shape}")


def test_PTE_with_independent_signals():
    signal_length = 1000
    s1 = np.random.normal(0, 1, signal_length)
    s2 = np.random.normal(0, 1, signal_length)

    # Combine signals into a matrix
    signals = np.vstack([s1, s2])

    phase = get_phase(signals)
    phase_inc = phase + np.pi
    binsize = get_binsize(phase_inc)
    d_phase = get_discretized_phase(phase_inc, binsize)

    pte_matrix = compute_PTE(d_phase, 1)

    # Check off-diagonal elements for significant PTE
    # (s1 -> s2 and s2 -> s1 should be low)
    assert np.isclose(pte_matrix[0, 1], 0, atol=1e-2) and np.isclose(
        pte_matrix[1, 0],
        0,
        atol=1e-2), "PTE between independent signals should be negligible."



