import numpy as np
import pytest

from pyPTE.core.pyPTE import (
    PTE,
    compute_PTE,
    get_bincount,
    get_binsize,
    get_delay,
    get_discretized_phase,
    get_phase,
)

FS = 250.0


def coupled_pair(seed=0, n=8000, lag=12, freq=10.0, flip=False):
    """Two channels where one drives the other by `lag` samples.

    Returns an (m_channels, n_samples) array. With ``flip=False`` channel 0 is
    the driver; with ``flip=True`` the channel order is reversed so the expected
    direction inverts.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    driver = np.sin(2 * np.pi * freq * t) + 0.2 * rng.standard_normal(n)
    target = 0.9 * np.roll(driver, lag) + 0.4 * rng.standard_normal(n)
    return np.vstack([target, driver]) if flip else np.vstack([driver, target])


def independent_pair(seed=0, n=8000):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, n))


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def test_get_phase_zeros_input():
    time_series = np.array([[0, 0, 0], [0, 0, 0]])
    np.testing.assert_almost_equal(
        get_phase(time_series), np.array([[0, 0, 0], [0, 0, 0]]), decimal=5
    )


def test_get_phase_of_sine_is_bounded_and_advances():
    t = np.arange(2000) / FS
    phase = get_phase(np.sin(2 * np.pi * 10 * t)[np.newaxis, :])
    assert phase.shape == (1, 2000)
    assert phase.min() >= -np.pi and phase.max() <= np.pi
    # a 10 Hz sine over 8 s must wrap through 2*pi once per cycle
    wraps = (np.diff(phase[0]) < -np.pi).sum()
    assert wraps == pytest.approx(80, abs=2)


def test_get_discretized_phase():
    phase_data = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    discretized_phase = get_discretized_phase(phase_data, np.pi / 4)
    np.testing.assert_array_equal(discretized_phase, np.array([1, 2, 3]))


def test_function_shapes():
    time_series = np.random.default_rng(0).random((4, 1500))

    phase = get_phase(time_series)
    assert phase.shape == (4, 1500)
    assert isinstance(get_delay(phase), int)
    assert isinstance(get_binsize(phase), float)
    assert get_discretized_phase(phase, get_binsize(phase)).shape == (4, 1500)

    dPTE, raw_PTE = PTE(time_series)
    assert dPTE.shape == (4, 4)
    assert raw_PTE.shape == (4, 4)


# --------------------------------------------------------------------------
# directionality - the reason this library exists
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_pte_detects_coupling_direction(seed):
    """Channel 0 drives channel 1, so information must flow 0 -> 1."""
    dPTE, raw_PTE = PTE(coupled_pair(seed))
    assert dPTE[0, 1] > 0.5, "driver -> target should exceed the 0.5 midpoint"
    assert dPTE[1, 0] < 0.5
    assert raw_PTE[0, 1] > raw_PTE[1, 0]


@pytest.mark.parametrize("seed", range(3))
def test_direction_flips_with_channel_order(seed):
    """Swapping the channel order must invert the detected direction.

    Guards against the transposed-index family of bugs: a PTE matrix that is
    accidentally read as its own transpose still passes a single-direction
    check, but cannot survive this one.
    """
    forward, _ = PTE(coupled_pair(seed, flip=False))
    reversed_, _ = PTE(coupled_pair(seed, flip=True))
    assert forward[0, 1] > 0.5 and reversed_[1, 0] > 0.5
    assert forward[0, 1] == pytest.approx(reversed_[1, 0], abs=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_independent_signals_have_no_preferred_direction(seed):
    """Independent noise carries no directionality, so dPTE sits at 0.5."""
    dPTE, _ = PTE(independent_pair(seed))
    assert dPTE[0, 1] == pytest.approx(0.5, abs=0.05)
    assert dPTE[1, 0] == pytest.approx(0.5, abs=0.05)


def test_chain_recovers_indirect_direction():
    """In a -> b -> c every downstream pair must point forwards."""
    rng = np.random.default_rng(7)
    n = 8000
    t = np.arange(n) / FS
    a = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.standard_normal(n)
    b = 0.9 * np.roll(a, 10) + 0.4 * rng.standard_normal(n)
    c = 0.9 * np.roll(b, 10) + 0.4 * rng.standard_normal(n)

    dPTE, _ = PTE(np.vstack([a, b, c]))
    assert dPTE[0, 1] > 0.5 and dPTE[1, 2] > 0.5 and dPTE[0, 2] > 0.5


# --------------------------------------------------------------------------
# information-theoretic invariants
# --------------------------------------------------------------------------

def test_raw_pte_is_non_negative():
    """PTE is a conditional mutual information, so it cannot be negative."""
    _, raw_PTE = PTE(coupled_pair(0))
    assert (raw_PTE >= -1e-9).all()


def test_raw_pte_respects_entropy_upper_bound():
    """PTE = I(Ypr; X | Y) <= H(Ypr) <= log2(n_bins).

    This is the invariant that both historical regressions violated: dropping
    the joint-histogram counts drove PTE to -H(Y), and indexing ``np.add.at``
    with a list instead of a tuple inflated the counts far past any
    probability-normalised value.
    """
    time_series = coupled_pair(0)
    phase = get_phase(time_series) + np.pi
    n_bins = get_bincount(get_binsize(phase))

    _, raw_PTE = PTE(time_series)
    assert (raw_PTE <= np.log2(n_bins)).all()
    assert (raw_PTE >= -1e-9).all()


def test_self_transfer_entropy_is_zero():
    """A channel predicts itself no better with itself as source."""
    _, raw_PTE = PTE(coupled_pair(0))
    np.testing.assert_allclose(np.diag(raw_PTE), 0.0, atol=1e-9)


def test_dpte_is_antisymmetric_about_half():
    dPTE, _ = PTE(coupled_pair(0, n=4000))
    m = dPTE.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            assert dPTE[i, j] + dPTE[j, i] == pytest.approx(1.0)
    np.testing.assert_allclose(np.diag(dPTE), 0.0)


def test_raw_pte_depends_on_the_source_channel():
    """Direct guard for the 2024 regression.

    When the joint histograms were allocated but never filled, ``PTE[i, j]``
    collapsed to exactly ``-H(Y_j)`` - identical in every row, so the matrix
    varied only with the target channel. Any implementation that carries real
    directional information must produce differing rows.
    """
    rng = np.random.default_rng(3)
    n = 6000
    t = np.arange(n) / FS
    a = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.standard_normal(n)
    b = 0.9 * np.roll(a, 10) + 0.4 * rng.standard_normal(n)
    c = rng.standard_normal(n)

    _, raw_PTE = PTE(np.vstack([a, b, c]))
    assert not np.allclose(raw_PTE, raw_PTE[0][np.newaxis, :].repeat(3, axis=0)), (
        "raw PTE does not vary with the source channel - joint histograms "
        "are almost certainly not being populated"
    )


def test_compute_PTE_counts_use_every_sample():
    """The plug-in estimator must see exactly n - delay observations.

    Indexing ``np.add.at`` with a list rather than a tuple silently multiplied
    the joint counts by the size of the trailing axes; the resulting entropies
    scale with the data length instead of staying bounded by log2(n_bins).
    """
    time_series = coupled_pair(0, n=4000)
    phase = get_phase(time_series)
    delay = get_delay(phase)
    phase_inc = phase + np.pi
    d_phase = get_discretized_phase(phase_inc, get_binsize(phase_inc))

    raw_PTE = compute_PTE(d_phase, delay)
    n_bins = get_bincount(get_binsize(phase_inc))

    assert np.isfinite(raw_PTE).all()
    assert (np.abs(raw_PTE) <= np.log2(n_bins)).all()
