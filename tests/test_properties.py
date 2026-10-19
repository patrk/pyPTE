"""Invariants that must hold for any input, checked on generated signals."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pyPTE import PTE, benjamini_hochberg

# the first call pays for imports and caching, which a deadline would punish
settings.register_profile("pypte", deadline=None, max_examples=60)
settings.load_profile("pypte")


@st.composite
def recordings(draw):
    """Bounded real-valued (m, n) arrays, including degenerate ones."""
    m = draw(st.integers(2, 4))
    n = draw(st.integers(64, 600))
    values = st.floats(-1e3, 1e3, allow_nan=False, allow_subnormal=False)
    return draw(arrays(np.float64, (m, n), elements=values))


@st.composite
def noise(draw):
    """Continuous random (m, n) signals.

    The relabelling and scaling checks need a phase that is defined everywhere.
    Where the analytic signal is numerically zero, as in long runs of exact
    zeros, its angle is rounding noise, and FFT rounding depends on a row's
    position in the batch.
    """
    m = draw(st.integers(2, 4))
    n = draw(st.integers(64, 600))
    seed = draw(st.integers(0, 2**32 - 1))
    return np.random.default_rng(seed).standard_normal((m, n))


bin_counts = st.integers(2, 16)
delays = st.integers(1, 20)


@given(recordings(), bin_counts, delays)
def test_raw_pte_lies_between_zero_and_log2_bins(data, n_bins, delay):
    _, raw = PTE(data, binning=n_bins, delay=delay)
    # conditional mutual information is never negative and never exceeds the
    # entropy of one variable with n_bins states
    assert np.all(raw >= 0)
    assert np.all(raw <= np.log2(n_bins) + 1e-9)


@given(recordings(), bin_counts, delays)
def test_diagonals_are_zero(data, n_bins, delay):
    dPTE, raw = PTE(data, binning=n_bins, delay=delay)
    assert np.all(np.diag(dPTE) == 0)
    assert np.all(np.diag(raw) == 0)


@given(recordings(), bin_counts, delays)
def test_dpte_is_a_share_of_both_directions(data, n_bins, delay):
    dPTE, _ = PTE(data, binning=n_bins, delay=delay)
    off = ~np.eye(len(dPTE), dtype=bool)
    # undefined exactly when there is no transfer either way, so symmetrically
    assert np.array_equal(np.isnan(dPTE), np.isnan(dPTE.T))
    defined = off & ~np.isnan(dPTE)
    assert np.all((dPTE[defined] >= 0) & (dPTE[defined] <= 1))
    np.testing.assert_allclose((dPTE + dPTE.T)[defined], 1.0)


@given(noise(), st.randoms(use_true_random=False))
def test_relabelling_channels_permutes_the_result(data, random):
    order = list(range(len(data)))
    random.shuffle(order)
    dPTE, raw = PTE(data, binning=8, delay=5)
    dPTE_p, raw_p = PTE(data[order], binning=8, delay=5)
    np.testing.assert_array_equal(raw_p, raw[np.ix_(order, order)])
    np.testing.assert_array_equal(dPTE_p, dPTE[np.ix_(order, order)])


@given(noise(), st.integers(-8, 8))
def test_scaling_by_a_power_of_two_changes_nothing(data, exponent):
    # phase ignores amplitude, and a power-of-two factor is exact in floating
    # point, so even Scott's rule and the delay estimate must agree bit for bit
    dPTE, raw = PTE(data)
    dPTE_s, raw_s = PTE(data * 2.0**exponent)
    np.testing.assert_array_equal(raw_s, raw)
    np.testing.assert_array_equal(dPTE_s, dPTE)


@given(
    arrays(np.float64, st.integers(1, 50), elements=st.floats(0, 1)),
    st.floats(0.001, 0.2),
    st.floats(0.001, 0.2),
)
def test_benjamini_hochberg_accepts_more_as_alpha_grows(p_values, a, b):
    low, high = sorted((a, b))
    kept_low, _ = benjamini_hochberg(p_values, low)
    kept_high, _ = benjamini_hochberg(p_values, high)
    assert np.all(kept_high[kept_low])
