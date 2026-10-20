import warnings

import numpy as np
import pytest

from pyPTE.stats import (
    benjamini_hochberg,
    surrogate_test,
    surrogates_needed,
    time_shifted_surrogate,
)

FS = 250.0


def oscillator(rng, n, freq=10.0):
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift)


def independent_pair(n=6000, noise_ratio=1.0, seed=7):
    """Two unrelated oscillators, optionally with very different noise levels."""
    rng = np.random.default_rng(seed)
    a = oscillator(rng, n) + 0.3 * rng.standard_normal(n)
    b = oscillator(rng, n) + 0.3 * noise_ratio * rng.standard_normal(n)
    return np.vstack([a, b])


def coupled_pair(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    driver = oscillator(rng, n) + 0.3 * rng.standard_normal(n)
    target = 0.9 * np.roll(driver, 12) + 0.3 * rng.standard_normal(n)
    return np.vstack([driver, target])


# --------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------


def test_bh_rejects_nothing_when_all_null():
    rng = np.random.default_rng(0)
    # p-values under a true null are uniform on [0, 1]
    p = rng.uniform(size=500)
    significant, threshold = benjamini_hochberg(p, alpha=0.05)
    assert significant.sum() <= 25, "FDR should not flood a pure null with hits"
    assert threshold >= 0.0


def test_bh_finds_obvious_signal():
    p = np.concatenate([np.full(10, 1e-6), np.linspace(0.2, 1.0, 490)])
    significant, threshold = benjamini_hochberg(p, alpha=0.05)
    assert significant[:10].all(), "clearly significant p-values must survive"
    assert threshold > 0


def test_bh_is_less_conservative_than_bonferroni():
    """The reason to prefer FDR on a connectivity matrix at all.

    Sized like a 100-channel montage: 9900 ordered pairs. Bonferroni's cutoff
    is then alpha/9900 = 5.05e-6, while Benjamini-Hochberg admits p-values up
    to alpha * 50 / 9900 = 2.5e-4 when 50 of them are genuinely small. The
    effects below sit deliberately between those two thresholds.
    """
    p = np.concatenate([np.full(50, 1e-5), np.linspace(0.2, 1.0, 9850)])
    significant, _ = benjamini_hochberg(p, alpha=0.05)
    bonferroni = p <= 0.05 / p.size

    assert significant[:50].all(), "FDR should recover these effects"
    assert not bonferroni[:50].any(), "Bonferroni is expected to miss them entirely"
    assert significant.sum() > bonferroni.sum()
    assert not significant[50:].any(), "and nothing beyond them"


def test_bh_preserves_shape():
    p = np.random.default_rng(0).uniform(size=(6, 6))
    significant, _ = benjamini_hochberg(p)
    assert significant.shape == (6, 6)


# --------------------------------------------------------------------------
# surrogates
# --------------------------------------------------------------------------


def test_surrogate_preserves_each_channel_exactly():
    """A circular shift must not alter any channel's marginal properties.

    This is the whole basis of the null: if the surrogate changed a channel's
    own distribution it would no longer isolate the cross-channel timing.
    """
    data = independent_pair(n=2000)
    surrogate = time_shifted_surrogate(data, np.random.default_rng(0))

    for channel in range(data.shape[0]):
        np.testing.assert_allclose(np.sort(surrogate[channel]), np.sort(data[channel]))
    assert not np.allclose(surrogate, data), "the alignment should have changed"


def test_surrogate_shifts_channels_independently():
    data = independent_pair(n=2000)
    surrogate = time_shifted_surrogate(data, np.random.default_rng(1))

    # if both channels moved by the same offset their relative timing, and so
    # the PTE between them, would be unchanged
    lag0 = int(np.argmax(np.correlate(surrogate[0], data[0], mode="same")))
    lag1 = int(np.argmax(np.correlate(surrogate[1], data[1], mode="same")))
    assert lag0 != lag1


# --------------------------------------------------------------------------
# the surrogate test end to end
# --------------------------------------------------------------------------


def test_unequal_snr_alone_is_not_significant():
    """The headline claim: surrogates cancel the SNR-induced false positive.

    Two independent oscillators with a 4x noise gap produce dPTE around 0.73.
    Read naively that looks like strong coupling; against the null it is not.
    """
    result = surrogate_test(independent_pair(noise_ratio=4.0), n_surrogates=100, seed=0)

    assert result.dPTE[0, 1] > 0.6, "the raw value should look coupled"
    assert not result.significant[0, 1], "but it must not survive the null"
    assert abs(result.null_mean[0, 1] - result.dPTE[0, 1]) < 0.1, (
        "the null should reproduce the bias, showing it is a per-channel property"
    )


def test_genuine_coupling_is_significant():
    result = surrogate_test(coupled_pair(), n_surrogates=100, seed=0)

    assert result.significant[0, 1], "real coupling should survive the null"
    assert result.p_values[0, 1] < 0.05


def test_independent_matched_signals_are_not_significant():
    result = surrogate_test(independent_pair(noise_ratio=1.0), n_surrogates=100, seed=0)
    assert not result.significant.any()


# these check plumbing rather than power, so a deliberately small
# surrogate count is fine and the underpowered warning is not the point
@pytest.mark.filterwarnings("ignore:.*cannot produce a p-value.*:RuntimeWarning")
def test_result_shapes_and_diagonal():
    data = np.random.default_rng(0).standard_normal((4, 2000))
    result = surrogate_test(data, n_surrogates=20, seed=0)

    assert result.dPTE.shape == (4, 4)
    assert result.p_values.shape == (4, 4)
    assert result.significant.shape == (4, 4)
    np.testing.assert_array_equal(np.diag(result.p_values), 1.0)
    assert not np.diag(result.significant).any()


def test_p_values_are_bounded_away_from_zero():
    """A finite null cannot support a p-value below 1 / (n + 1)."""
    data = coupled_pair()
    n_surrogates = 50
    result = surrogate_test(data, n_surrogates=n_surrogates, seed=0)

    assert (result.p_values >= 1.0 / (n_surrogates + 1)).all()
    assert (result.p_values <= 1.0).all()


# these check plumbing rather than power, so a deliberately small
# surrogate count is fine and the underpowered warning is not the point
@pytest.mark.filterwarnings("ignore:.*cannot produce a p-value.*:RuntimeWarning")
def test_null_distribution_is_optional():
    data = coupled_pair(n=2000)

    without = surrogate_test(data, n_surrogates=10, seed=0)
    assert without.null_distribution is None

    with_null = surrogate_test(data, n_surrogates=10, seed=0, keep_null=True)
    assert with_null.null_distribution is not None
    assert with_null.null_distribution.shape == (10, 2, 2)
    # the retained surrogates must be the ones the summary statistics came from
    np.testing.assert_allclose(
        with_null.null_distribution.mean(axis=0), with_null.null_mean
    )
    np.testing.assert_allclose(with_null.p_values, without.p_values)


# these check plumbing rather than power, so a deliberately small
# surrogate count is fine and the underpowered warning is not the point
@pytest.mark.filterwarnings("ignore:.*cannot produce a p-value.*:RuntimeWarning")
def test_seed_makes_it_reproducible():
    data = coupled_pair()
    first = surrogate_test(data, n_surrogates=20, seed=42)
    second = surrogate_test(data, n_surrogates=20, seed=42)
    np.testing.assert_allclose(first.p_values, second.p_values)


# these check plumbing rather than power, so a deliberately small
# surrogate count is fine and the underpowered warning is not the point
@pytest.mark.filterwarnings("ignore:.*cannot produce a p-value.*:RuntimeWarning")
def test_summary_mentions_the_correction():
    result = surrogate_test(coupled_pair(), n_surrogates=20, seed=0)
    assert "Benjamini-Hochberg" in result.summary()
    assert str(result.n_surrogates) in result.summary()


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="m x n"):
        surrogate_test(np.zeros(10), n_surrogates=5)
    with pytest.raises(ValueError, match="at least 1"):
        surrogate_test(np.zeros((2, 100)), n_surrogates=0)


# --------------------------------------------------------------------------
# power arithmetic
# --------------------------------------------------------------------------


def test_surrogates_needed_scales_with_pairs():
    """The floor is alpha / n_pairs, so the requirement grows quadratically."""
    assert surrogates_needed(2) < surrogates_needed(20) < surrogates_needed(76)
    # 76 channels is 2850 unordered pairs; p[i, j] == p[j, i], so each pair
    # enters the correction twice and is accepted at rank 2: p <= 0.05 / 2850
    assert surrogates_needed(76) == pytest.approx(2850 / 0.05 - 1, rel=1e-6)


def test_surrogates_needed_handles_degenerate_input():
    assert surrogates_needed(1) == 1
    assert surrogates_needed(0) == 1


def test_warns_when_no_edge_could_possibly_survive():
    """The trap this exists to catch: a null result that was never reachable."""
    data = np.random.default_rng(0).standard_normal((30, 500))

    with pytest.warns(RuntimeWarning, match="cannot produce a p-value below"):
        result = surrogate_test(data, n_surrogates=10, seed=0)

    assert not result.significant.any()


def test_no_warning_when_the_count_is_adequate():
    data = independent_pair(n=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        surrogate_test(data, n_surrogates=200, seed=0)


def test_warning_names_a_usable_alternative():
    data = np.random.default_rng(0).standard_normal((30, 500))
    with pytest.warns(RuntimeWarning) as caught:
        surrogate_test(data, n_surrogates=10, seed=0)

    message = str(caught[0].message)
    assert "cluster_permutation" in message
    assert str(surrogates_needed(30)) in message


@pytest.mark.parametrize("m", [2, 3, 5, 10])
def test_surrogates_needed_is_the_exact_floor(m):
    """At the returned count an isolated perfect edge survives; one fewer, not."""
    from pyPTE.stats import _correct_off_diagonal

    def survives(n_surrogates):
        p = np.ones((m, m))
        p[0, 1] = p[1, 0] = 1.0 / (n_surrogates + 1)
        return bool(_correct_off_diagonal(p, 0.05)[0][0, 1])

    n = surrogates_needed(m)
    assert survives(n)
    assert not survives(n - 1)


def test_minimum_surrogate_count_detects_strong_coupling():
    """End to end: the count surrogates_needed asks for must be enough."""
    n = surrogates_needed(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = surrogate_test(coupled_pair(), n_surrogates=n, seed=0)
    assert result.significant[0, 1] and result.significant[1, 0]


def test_correction_ignores_the_diagonal():
    """The diagonal is not a test; counting it makes the correction stricter."""
    from pyPTE.stats import _correct_off_diagonal

    m = 4
    p = np.ones((m, m))
    # passes BH over the 12 off-diagonal entries at rank 2, fails over all 16
    p[0, 1] = p[1, 0] = 0.05 * 2 / 12
    significant, _ = _correct_off_diagonal(p, 0.05)
    assert significant[0, 1] and significant[1, 0]
    assert not significant[np.eye(m, dtype=bool)].any()


def test_nan_dpte_is_never_significant():
    """A duplicated channel has zero PTE both ways, so dPTE is 0/0."""
    data = coupled_pair()
    data = np.vstack([data, data[0]])
    with pytest.warns(RuntimeWarning, match="undefined"):
        result = surrogate_test(data, n_surrogates=surrogates_needed(3), seed=0)
    assert result.p_values[0, 2] == 1.0 and result.p_values[2, 0] == 1.0
    assert not result.significant[0, 2] and not result.significant[2, 0]


def test_surrogate_test_forwards_binning_and_delay():
    data = coupled_pair()
    from pyPTE import PTE

    result = surrogate_test(data, n_surrogates=5, seed=0, binning=6, delay=10)
    expected, _ = PTE(data, binning=6, delay=10)
    np.testing.assert_allclose(result.dPTE, expected)


def test_surrogates_reuse_the_observed_delay_and_bins(monkeypatch):
    # with the default rules the delay and bin count are estimated from data,
    # so a surrogate re-estimating them would be tested under other settings
    import pyPTE.core.pyPTE as core
    import pyPTE.stats as stats

    phases, settings = [], []
    real_phase, real_counts = core.get_phase, core.compute_dPTE_rawPTE

    def spy_phase(*args, **kwargs):
        phases.append(1)
        return real_phase(*args, **kwargs)

    def spy_counts(d_phase, delay, n_bins=None, **kwargs):
        settings.append((delay, n_bins))
        return real_counts(d_phase, delay, n_bins=n_bins, **kwargs)

    monkeypatch.setattr(core, "get_phase", spy_phase)
    monkeypatch.setattr(stats, "compute_dPTE_rawPTE", spy_counts)

    surrogate_test(coupled_pair(), n_surrogates=20, seed=0)

    assert len(phases) == 1
    assert len(settings) == 21 and len(set(settings)) == 1


def test_surrogate_test_trims_like_pte():
    from pyPTE import PTE

    data = coupled_pair()
    result = surrogate_test(data, n_surrogates=5, seed=0, trim_cycles=2)
    expected, _ = PTE(data, trim_cycles=2)
    np.testing.assert_array_equal(result.dPTE, expected)


def test_independent_signals_give_roughly_uniform_p_values():
    # the null must stay calibrated now that surrogates shift the binned phase
    p = [
        surrogate_test(independent_pair(seed=s), n_surrogates=99, seed=s).p_values[0, 1]
        for s in range(60)
    ]
    assert 0.02 <= np.mean(np.asarray(p) <= 0.1) <= 0.25
    assert 0.35 <= np.mean(p) <= 0.65


def test_parallel_surrogates_match_serial_ones():
    data = coupled_pair()
    serial = surrogate_test(data, n_surrogates=19, seed=0, keep_null=True)
    parallel = surrogate_test(data, n_surrogates=19, seed=0, keep_null=True, n_jobs=2)
    np.testing.assert_array_equal(parallel.p_values, serial.p_values)
    np.testing.assert_array_equal(parallel.null_distribution, serial.null_distribution)
    np.testing.assert_array_equal(parallel.effective_PTE, serial.effective_PTE)


def test_n_jobs_must_be_positive_or_minus_one():
    with pytest.raises(ValueError, match="n_jobs"):
        surrogate_test(coupled_pair(), n_surrogates=5, n_jobs=0)


# --------------------------------------------------------------------------
# effective PTE
# --------------------------------------------------------------------------


def test_effective_pte_removes_the_bias_floor_of_independent_signals():
    result = surrogate_test(independent_pair(), n_surrogates=50, seed=0)
    off = ~np.eye(2, dtype=bool)
    # the plug-in estimate sits well above zero with no coupling at all ...
    assert np.all(result.raw_PTE[off] > 0.01)
    # ... and almost all of that is the bias the surrogates also carry
    assert np.all(np.abs(result.effective_PTE[off]) < 0.2 * result.raw_PTE[off])


def test_effective_pte_keeps_genuine_transfer():
    result = surrogate_test(coupled_pair(), n_surrogates=50, seed=0)
    assert result.effective_PTE[0, 1] > 0.05
    assert result.effective_PTE[0, 1] > 5 * abs(result.effective_PTE[1, 0])


def test_effective_pte_is_raw_pte_minus_the_raw_null_mean():
    result = surrogate_test(coupled_pair(), n_surrogates=20, seed=0)
    expected = result.raw_PTE - result.raw_null_mean
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_array_equal(result.effective_PTE, expected)
    assert np.all(np.diag(result.effective_PTE) == 0)


def test_effective_pte_is_not_clipped_at_zero():
    # averaging over subjects needs the below-chance values, or the mean of
    # pure noise would come out positive
    result = surrogate_test(independent_pair(), n_surrogates=50, seed=0, binning=4)
    off = ~np.eye(2, dtype=bool)
    assert result.effective_PTE[off].min() < 0 < result.raw_PTE[off].min()


def test_effective_pte_is_comparable_across_bin_counts():
    data = independent_pair()
    off = ~np.eye(2, dtype=bool)
    raw, effective = [], []
    for n_bins in (4, 8, 16):
        result = surrogate_test(data, n_surrogates=50, seed=0, binning=n_bins)
        raw.append(result.raw_PTE[off])
        effective.append(result.effective_PTE[off])
    # the bin count moves raw PTE by more than a tenth of a bit with no
    # coupling present; the effective value barely moves
    assert np.ptp(raw) > 0.1
    assert np.ptp(effective) < 0.25 * np.ptp(raw)
