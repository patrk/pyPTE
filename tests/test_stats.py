import numpy as np
import pytest

from pyPTE.stats import benjamini_hochberg, surrogate_test, time_shifted_surrogate

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


def test_seed_makes_it_reproducible():
    data = coupled_pair()
    first = surrogate_test(data, n_surrogates=20, seed=42)
    second = surrogate_test(data, n_surrogates=20, seed=42)
    np.testing.assert_allclose(first.p_values, second.p_values)


def test_summary_mentions_the_correction():
    result = surrogate_test(coupled_pair(), n_surrogates=20, seed=0)
    assert "Benjamini-Hochberg" in result.summary()
    assert str(result.n_surrogates) in result.summary()


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="m x n"):
        surrogate_test(np.zeros(10), n_surrogates=5)
    with pytest.raises(ValueError, match="at least 1"):
        surrogate_test(np.zeros((2, 100)), n_surrogates=0)
