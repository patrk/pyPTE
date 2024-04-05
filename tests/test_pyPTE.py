import numpy as np
from pyPTE.core.pyPTE import get_phase, get_binsize, get_discretized_phase
from pyPTE.core.pyPTE import compute_PTE, PTE

def test_get_phase_zeros_input():
    time_series = np.array([[0, 0, 0], [0, 0, 0]])
    expected_phase_result = np.array([[0, 0, 0], [0, 0, 0]])
    np.testing.assert_almost_equal(get_phase(time_series), expected_phase_result, decimal=5)

def test_get_discretized_phase():
    phase_data = np.array([np.pi/4, np.pi/2, 3*np.pi/4])
    binsize = np.pi/4
    expected_discretized_phase = np.array([1, 2, 3])
    discretized_phase = get_discretized_phase(phase_data, binsize)
    np.testing.assert_array_equal(discretized_phase, expected_discretized_phase, "Discretized phase did not match expected values.")

def test_PTE_integration():
    time_series = np.random.rand(5, 100) 
    dPTE, raw_PTE = PTE(time_series)
    assert dPTE.shape == (5, 5), f"Expected dPTE shape (5, 5), got {dPTE.shape}"
    assert raw_PTE.shape == (5, 5), f"Expected raw_PTE shape (5, 5), got {raw_PTE.shape}"

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

    # Check off-diagonal elements for significant PTE (s1 -> s2 and s2 -> s1 should be low)
    assert np.isclose(pte_matrix[0, 1], 0, atol=1e-2) and np.isclose(pte_matrix[1, 0], 0, atol=1e-2), "PTE between independent signals should be negligible."



