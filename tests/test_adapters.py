import numpy as np
import pytest

mne = pytest.importorskip("mne")
pd = pytest.importorskip("pandas")

from pyPTE.adapters.mne_adapter import PTE_from_mne, interpolate_mne  # noqa: E402
from pyPTE.adapters.pandas_adapter import PTE_from_dataframe  # noqa: E402

FS = 250.0
CHANNELS = ["driver", "target", "noise"]


def coupled_signals(n=6000, lag=12, seed=0):
    """Channel 0 drives channel 1; channel 2 is unrelated noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    driver = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.standard_normal(n)
    target = 0.9 * np.roll(driver, lag) + 0.4 * rng.standard_normal(n)
    noise = rng.standard_normal(n)
    return np.vstack([driver, target, noise])


def make_raw(data, ch_names=CHANNELS, ch_type="eeg"):
    info = mne.create_info(list(ch_names), sfreq=FS, ch_types=ch_type)
    return mne.io.RawArray(data, info, verbose=False)


# --------------------------------------------------------------------------
# pandas adapter
# --------------------------------------------------------------------------


def test_dataframe_adapter_recovers_direction():
    """A time-major frame must yield the same directionality as the raw array."""
    data = coupled_signals()
    frame = pd.DataFrame(data.T, columns=CHANNELS)

    dPTE, raw_PTE = PTE_from_dataframe(frame)

    assert list(dPTE.index) == CHANNELS
    assert list(dPTE.columns) == CHANNELS
    assert dPTE.loc["driver", "target"] > 0.5
    assert dPTE.loc["target", "driver"] < 0.5
    assert raw_PTE.loc["driver", "target"] > raw_PTE.loc["target", "driver"]


def test_dataframe_adapter_matches_core():
    """The adapter must not alter the numbers, only label them."""
    from pyPTE import PTE

    data = coupled_signals()
    expected_dPTE, expected_raw = PTE(data)

    dPTE, raw_PTE = PTE_from_dataframe(pd.DataFrame(data.T, columns=CHANNELS))

    np.testing.assert_allclose(dPTE.to_numpy(), expected_dPTE)
    np.testing.assert_allclose(raw_PTE.to_numpy(), expected_raw)


def test_dataframe_adapter_is_orientation_sensitive():
    """Feeding a channel-major frame must not silently produce the same answer.

    The frame is documented as time-major. Transposing it makes each time point
    look like a channel, which has to change the result - otherwise the adapter
    is ignoring orientation, which is how it was broken before.
    """
    data = coupled_signals(n=400)
    correct, _ = PTE_from_dataframe(pd.DataFrame(data.T, columns=CHANNELS))
    swapped, _ = PTE_from_dataframe(pd.DataFrame(data))

    assert correct.shape == (3, 3)
    assert swapped.shape == (400, 400)


# --------------------------------------------------------------------------
# mne adapter
# --------------------------------------------------------------------------


def test_mne_adapter_recovers_direction():
    raw = make_raw(coupled_signals())

    dPTE, raw_PTE = PTE_from_mne(raw)

    assert list(dPTE.index) == CHANNELS
    assert dPTE.loc["driver", "target"] > 0.5
    assert dPTE.loc["target", "driver"] < 0.5


def test_mne_adapter_matches_core():
    """Going through mne must not change the numbers."""
    from pyPTE import PTE

    data = coupled_signals()
    expected_dPTE, _ = PTE(data)

    dPTE, _ = PTE_from_mne(make_raw(data))

    np.testing.assert_allclose(dPTE.to_numpy(), expected_dPTE)


def test_mne_adapter_respects_picks():
    raw = make_raw(coupled_signals())

    dPTE, _ = PTE_from_mne(raw, picks=["driver", "target"])

    assert list(dPTE.index) == ["driver", "target"]
    assert dPTE.shape == (2, 2)
    assert dPTE.loc["driver", "target"] > 0.5


def test_mne_adapter_does_not_mutate_input():
    """picks must not strip channels from the caller's object."""
    raw = make_raw(coupled_signals())

    PTE_from_mne(raw, picks=["driver", "target"])

    assert raw.ch_names == CHANNELS


# --------------------------------------------------------------------------
# channel interpolation
# --------------------------------------------------------------------------


def with_montage(raw):
    """Attach positions so interpolate_bads() has a spatial model to work with."""
    montage = mne.channels.make_standard_montage("standard_1020")
    return raw.set_montage(montage, verbose=False)


# enough electrodes, spread over the scalp, for a sane spherical head fit
MONTAGE_NAMES = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]


def test_interpolate_restores_missing_channels():
    rng = np.random.default_rng(0)
    missing = ["O1", "O2"]
    present = [ch for ch in MONTAGE_NAMES if ch not in missing]

    reference = with_montage(
        make_raw(rng.standard_normal((len(MONTAGE_NAMES), 1000)), MONTAGE_NAMES)
    )
    partial = with_montage(make_raw(rng.standard_normal((len(present), 1000)), present))

    restored = interpolate_mne(partial, reference)

    assert set(restored.ch_names) == set(MONTAGE_NAMES)
    # the placeholders start as zeros, so non-zero output proves interpolation ran
    assert np.any(restored.copy().pick(missing).get_data() != 0.0)


def test_interpolate_is_a_noop_when_nothing_is_missing():
    names = ["Fp1", "Fp2", "C3"]
    raw = with_montage(
        make_raw(np.random.default_rng(0).standard_normal((3, 500)), names)
    )

    restored = interpolate_mne(raw, raw)

    assert restored.ch_names == raw.ch_names
    np.testing.assert_allclose(restored.get_data(), raw.get_data())


def test_interpolate_rejects_mismatched_sampling_rates():
    names = ["Fp1", "Fp2", "C3"]
    rng = np.random.default_rng(0)
    raw = with_montage(make_raw(rng.standard_normal((2, 500)), names[:2]))

    info = mne.create_info(names, sfreq=FS * 2, ch_types="eeg")
    reference = mne.io.RawArray(rng.standard_normal((3, 500)), info, verbose=False)
    reference = with_montage(reference)

    with pytest.raises(ValueError, match="sampling frequency"):
        interpolate_mne(raw, reference)
