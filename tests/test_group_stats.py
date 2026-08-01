import numpy as np
import pytest

from pyPTE import PTE, group_contrast, group_test

FS = 250.0


def oscillator(rng, n, freq=10.0):
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift)


def epoch(n, coupling, seed):
    """One short epoch; channel 0 drives channel 1 when coupling > 0."""
    rng = np.random.default_rng(seed)
    driver = oscillator(rng, n) + 0.3 * rng.standard_normal(n)
    target = (
        coupling * np.roll(driver, 12)
        + max(1.0 - coupling, 0.0) * oscillator(rng, n)
        + 0.3 * rng.standard_normal(n)
    )
    return np.vstack([driver, target])


def matrices(n_epochs, n_samples, coupling, offset=0):
    return np.array(
        [PTE(epoch(n_samples, coupling, s + offset))[0] for s in range(n_epochs)]
    )


# --------------------------------------------------------------------------
# group_test
# --------------------------------------------------------------------------


def test_short_epochs_beat_a_single_long_recording():
    """The reason this exists: consistency across epochs is very powerful.

    Eighty epochs of a third of a second - 26 seconds of data in total -
    resolve a coupling that a single continuous minute leaves marginal under
    the surrogate test.
    """
    result = group_test(matrices(80, 80, 0.9))

    assert result.significant[0, 1]
    assert result.p_values[0, 1] < 1e-6
    assert result.effect[0, 1] > 0.5


def test_independent_epochs_are_not_significant():
    result = group_test(matrices(80, 80, 0.0))

    assert not result.significant.any()
    assert result.p_values[0, 1] > 0.05


def test_ttest_agrees_with_wilcoxon_on_a_clear_effect():
    data = matrices(40, 250, 0.9)

    wilcoxon = group_test(data, method="wilcoxon")
    ttest = group_test(data, method="ttest")

    assert wilcoxon.significant[0, 1] and ttest.significant[0, 1]
    np.testing.assert_allclose(wilcoxon.effect, ttest.effect)


def test_reference_can_be_moved_for_raw_pte():
    """Raw PTE is tested against 0, not 0.5."""
    raw = np.array([PTE(epoch(250, 0.9, s))[1] for s in range(20)])
    result = group_test(raw, reference=0.0)

    assert result.significant[0, 1]
    assert result.effect[0, 1] > 0


def test_diagonal_is_never_significant():
    result = group_test(matrices(20, 250, 0.9))

    assert not np.diag(result.significant).any()
    np.testing.assert_array_equal(np.diag(result.p_values), 1.0)


def test_group_test_rejects_bad_input():
    with pytest.raises(ValueError, match="n_observations"):
        group_test(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="at least 2"):
        group_test(np.zeros((1, 3, 3)))


# --------------------------------------------------------------------------
# group_contrast
# --------------------------------------------------------------------------


def test_contrast_detects_a_condition_difference():
    coupled = matrices(40, 250, 0.9)
    uncoupled = matrices(40, 250, 0.0)

    result = group_contrast(coupled, uncoupled)

    assert result.significant[0, 1]
    assert result.effect[0, 1] > 0, "condition A should show more A -> B flow"


def test_contrast_finds_nothing_between_equivalent_conditions():
    """Two samples of the same process must not differ."""
    result = group_contrast(matrices(40, 250, 0.9), matrices(40, 250, 0.9, offset=500))
    assert not result.significant.any()


def test_contrast_is_antisymmetric_in_its_arguments():
    a = matrices(30, 250, 0.9)
    b = matrices(30, 250, 0.0)

    forward = group_contrast(a, b)
    reverse = group_contrast(b, a)

    np.testing.assert_allclose(forward.effect, -reverse.effect)
    np.testing.assert_allclose(forward.p_values, reverse.p_values)


def test_contrast_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="matching shapes"):
        group_contrast(np.zeros((5, 3, 3)), np.zeros((4, 3, 3)))
    with pytest.raises(ValueError, match="at least 2"):
        group_contrast(np.zeros((1, 3, 3)), np.zeros((1, 3, 3)))


def test_summary_reports_the_method_and_count():
    result = group_contrast(matrices(20, 250, 0.9), matrices(20, 250, 0.0))
    assert "paired wilcoxon" in result.summary()
    assert "20 observations" in result.summary()
