"""The time-reversal-differenced dPTE, and the segment test that makes it robust.

dPTE on its own reports a direction for a difference in signal-to-noise ratio
or for one source mixed into two channels (#97). Reversing time flips a real
direction and leaves those artefacts in place, so the difference keeps the
first and cancels the second.
"""

import warnings

import numpy as np
import pytest

from pyPTE import PTE, PTE_segments, group_test, surrogate_test
from pyPTE.core.pyPTE import (
    compute_dPTE_rawPTE,
    compute_segments,
    prepare_phase,
    reversal_differenced_dPTE,
    reverse_phase,
)
from tests.signals import band_pass, coupled, mixed_pair, noisy_driver_pair

OFF = ~np.eye(2, dtype=bool)


def test_differenced_dpte_is_antisymmetric_about_a_half():
    differenced, _ = PTE(band_pass(noisy_driver_pair(n=8000)), time_reversal=True)
    np.testing.assert_allclose((differenced + differenced.T)[OFF], 1.0)
    assert np.all(np.diag(differenced) == 0)


def test_differenced_dpte_follows_the_formula():
    data = band_pass(noisy_driver_pair(n=8000))
    d_phase, delay, n_bins = prepare_phase(data)
    forward, raw = compute_dPTE_rawPTE(d_phase, delay, n_bins=n_bins)
    backward, _ = compute_dPTE_rawPTE(reverse_phase(d_phase), delay, n_bins=n_bins)

    differenced, differenced_raw = PTE(data, time_reversal=True)

    np.testing.assert_array_equal(
        differenced, reversal_differenced_dPTE(forward, backward)
    )
    # raw PTE stays the forward value; there is no differenced raw PTE
    np.testing.assert_array_equal(differenced_raw, raw)


def test_differencing_cancels_a_direction_made_by_snr():
    # unfiltered, a clean target looks like it drives its noisy driver
    data = noisy_driver_pair()
    plain, _ = PTE(data)
    differenced, _ = PTE(data, time_reversal=True)
    assert plain[1, 0] > 0.8
    assert abs(differenced[0, 1] - 0.5) < 0.02


def test_differencing_recovers_a_direction_that_plain_dpte_gets_backwards():
    # band-passed, a driver this noisy still looks driven to plain dPTE
    data = band_pass(noisy_driver_pair(driver_noise=2.0))
    plain, _ = PTE(data)
    differenced, _ = PTE(data, time_reversal=True)
    assert plain[0, 1] < 0.5
    assert differenced[0, 1] > 0.5


def test_differencing_cancels_instantaneous_mixing():
    plain, _ = PTE(mixed_pair())
    differenced, _ = PTE(mixed_pair(), time_reversal=True)
    assert abs(plain[0, 1] - 0.5) > 0.2
    assert abs(differenced[0, 1] - 0.5) < 0.02


# --------------------------------------------------------------------------
# segments
# --------------------------------------------------------------------------


def test_segments_share_one_phase_delay_and_bin_count():
    data = band_pass(noisy_driver_pair(n=8000))
    d_phase, delay, n_bins = prepare_phase(data, n_segments=4)
    expected, expected_raw = compute_segments(
        np.array_split(d_phase, 4, axis=1), delay, n_bins
    )

    dPTE, raw = PTE_segments(data, 4)

    assert dPTE.shape == raw.shape == (4, 2, 2)
    np.testing.assert_array_equal(dPTE, expected)
    np.testing.assert_array_equal(raw, expected_raw)


def test_segments_rejects_too_few_or_too_short_segments():
    data = band_pass(noisy_driver_pair(n=2000))
    with pytest.raises(ValueError, match="at least 2"):
        PTE_segments(data, 1)
    with pytest.raises(ValueError, match="too short"):
        PTE_segments(data, 500)


def test_segment_test_finds_a_genuine_direction():
    dPTE, _ = PTE_segments(band_pass(noisy_driver_pair()), 10, time_reversal=True)
    result = group_test(dPTE)
    assert result.significant[0, 1] and result.effect[0, 1] > 0.5


def test_segment_test_rejects_a_direction_made_by_snr():
    data = noisy_driver_pair()
    # the surrogate test calls the backwards direction significant ...
    assert surrogate_test(data, n_surrogates=99, seed=0).significant[1, 0]
    # ... and the differenced segments show nothing either way
    dPTE, _ = PTE_segments(data, 10, time_reversal=True)
    assert not group_test(dPTE).significant.any()


def test_segment_test_rejects_instantaneous_mixing():
    data = band_pass(mixed_pair())
    assert surrogate_test(data, n_surrogates=99, seed=0).significant.any()
    dPTE, _ = PTE_segments(data, 10, time_reversal=True)
    assert not group_test(dPTE).significant.any()


def test_surrogate_test_reports_the_differenced_effect():
    data = band_pass(coupled(np.random.default_rng(0), 8000))
    result = surrogate_test(data, n_surrogates=19, seed=0, time_reversal=True)
    assert result.reversed_dPTE is not None
    np.testing.assert_array_equal(
        result.differenced_dPTE,
        reversal_differenced_dPTE(result.dPTE, result.reversed_dPTE),
    )
    assert surrogate_test(data, n_surrogates=19, seed=0).differenced_dPTE is None


# --------------------------------------------------------------------------
# group_test power warning
# --------------------------------------------------------------------------


def test_group_test_warns_when_too_few_observations_could_be_significant():
    dPTE, _ = PTE_segments(band_pass(noisy_driver_pair()), 5, time_reversal=True)
    # one pair at alpha 0.05 needs 2 / 2**n <= 0.05, so six observations
    with pytest.warns(RuntimeWarning, match="at least 6"):
        group_test(dPTE)


def test_group_test_is_quiet_with_enough_observations():
    dPTE, _ = PTE_segments(band_pass(noisy_driver_pair()), 10, time_reversal=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        group_test(dPTE)


def test_group_test_warning_counts_raw_pte_both_ways():
    _, raw = PTE_segments(band_pass(noisy_driver_pair()), 7)
    # 2 ordered tests at alpha 0.05 need 2 / 2**n <= 0.025: 7 is enough
    with pytest.warns(RuntimeWarning, match="at least 7"):
        group_test(raw[:6], reference=0.0)


# --------------------------------------------------------------------------
# too few bins
# --------------------------------------------------------------------------


def test_too_few_bins_make_the_direction_depend_on_the_delay():
    # one-second segments get about 7 bins, where a real coupling's direction
    # follows the delay rather than the data (#99)
    data = band_pass(coupled(np.random.default_rng(0), 250 * 120, lag=12))
    directions = {
        delay: PTE_segments(data, 120, binning=7, delay=delay)[0][:, 0, 1].mean()
        for delay in (12, 13)
    }
    assert directions[12] > 0.6 and directions[13] < 0.4
    # with enough bins the delay no longer decides it
    fine = [
        PTE_segments(data, 8, binning=20, delay=delay)[0][:, 0, 1].mean()
        for delay in (12, 13)
    ]
    assert min(fine) > 0.5


def test_few_bins_warn_with_the_window_length_needed():
    data = band_pass(coupled(np.random.default_rng(0), 500))
    with pytest.warns(RuntimeWarning, match="windows of at least"):
        PTE(data)
    with pytest.warns(RuntimeWarning, match="set binning to 16"):
        PTE(data, binning=8)


def test_enough_bins_do_not_warn():
    data = band_pass(coupled(np.random.default_rng(0), 30000))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", ".*phase bins", RuntimeWarning)
        PTE(data)
        PTE(data[:, :2000], binning=16)
