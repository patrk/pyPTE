import numpy as np
import pytest

mne = pytest.importorskip("mne")
pd = pytest.importorskip("pandas")

from pyPTE.adapters.mne_adapter import PTE_from_mne, interpolate_mne  # noqa: E402
from pyPTE.adapters.pandas_adapter import PTE_from_dataframe  # noqa: E402
from tests.signals import band_pass, coupled  # noqa: E402

FS = 250.0
CHANNELS = ["driver", "target", "noise"]


def coupled_signals(n=6000, lag=12, seed=0):
    """Channel 0 drives channel 1; channel 2 is unrelated noise.

    Band-passed and with equal noise on the coupled pair, so the direction
    comes from the lag rather than from which channel is cleaner.
    """
    rng = np.random.default_rng(seed)
    pair = coupled(rng, n, lag=lag)
    return band_pass(np.vstack([pair, rng.standard_normal(n)]))


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
    # only the shapes matter, and three samples per "channel" are too few to
    # band-pass
    data = np.random.default_rng(0).standard_normal((3, 400))
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


# --------------------------------------------------------------------------
# parameter passthrough
# --------------------------------------------------------------------------


def test_adapters_forward_pte_parameters():
    from pyPTE import PTE

    data = coupled_signals()
    expected, _ = PTE(data, binning=6, delay=10, trim_cycles=1)

    from_frame, _ = PTE_from_dataframe(
        pd.DataFrame(data.T, columns=CHANNELS), binning=6, delay=10, trim_cycles=1
    )
    from_mne, _ = PTE_from_mne(make_raw(data), binning=6, delay=10, trim_cycles=1)

    np.testing.assert_allclose(from_frame.to_numpy(), expected)
    np.testing.assert_allclose(from_mne.to_numpy(), expected)


def test_multi_process_forwards_parameters_quietly(capsys):
    from pyPTE import PTE
    from pyPTE.core.handle_multi import multi_process

    data = coupled_signals(n=2000)
    results = multi_process({"a": data}, binning=6, delay=10, trim_cycles=1, n_jobs=1)

    expected, _ = PTE(data, binning=6, delay=10, trim_cycles=1)
    np.testing.assert_allclose(results[0]["a"][0], expected)
    assert capsys.readouterr().out == ""


def test_multi_process_rejects_a_zero_worker_count():
    from pyPTE.core.handle_multi import multi_process

    with pytest.raises(ValueError, match="n_jobs"):
        multi_process({"a": coupled_signals(n=500)}, n_jobs=0)


# --------------------------------------------------------------------------
# epochs
# --------------------------------------------------------------------------


def make_epochs(n_epochs=20, n_times=4000, seed=0):
    """Band-passed epochs with equal, realistic noise on the coupled pair.

    With matched signal-to-noise ratios a short epoch carries little
    directional information, so these are long: 16 s each.
    """
    rng = np.random.default_rng(seed)
    epochs = [
        band_pass(
            np.vstack(
                [
                    coupled(rng, n_times, lag=5, driver_noise=1.0, target_noise=1.0),
                    rng.standard_normal(n_times),
                ]
            )
        )
        for _ in range(n_epochs)
    ]
    info = mne.create_info(CHANNELS, sfreq=FS, ch_types="eeg")
    return mne.EpochsArray(np.array(epochs), info, verbose=False)


def test_epochs_give_one_matrix_per_epoch_ready_for_group_tests():
    from pyPTE import group_test
    from pyPTE.adapters.mne_adapter import PTE_from_epochs

    dPTE, raw = PTE_from_epochs(make_epochs())

    assert dPTE.shape == raw.shape == (20, 3, 3)
    result = group_test(dPTE)
    assert result.significant[0, 1] and result.effect[0, 1] > 0.5


def test_epochs_share_one_delay_and_bin_count():
    from pyPTE.adapters.mne_adapter import PTE_from_epochs
    from pyPTE.core.pyPTE import compute_dPTE_rawPTE, prepare_phase

    epochs = make_epochs(n_epochs=6)
    data = epochs.get_data()
    stacked, delay, n_bins = prepare_phase(data.reshape(-1, data.shape[2]))
    first, _ = compute_dPTE_rawPTE(stacked[:3], delay, n_bins=n_bins)

    dPTE, _ = PTE_from_epochs(epochs)
    np.testing.assert_array_equal(dPTE[0], first)


def test_epochs_respect_picks_and_parameters():
    from pyPTE.adapters.mne_adapter import PTE_from_epochs

    epochs = make_epochs(n_epochs=4)
    dPTE, _ = PTE_from_epochs(epochs, picks=["driver", "target"], binning=6, delay=10)
    assert dPTE.shape == (4, 2, 2)


def test_epochs_can_be_time_reversal_differenced():
    from pyPTE import group_test
    from pyPTE.adapters.mne_adapter import PTE_from_epochs
    from pyPTE.core.pyPTE import (
        compute_dPTE_rawPTE,
        prepare_phase,
        reversal_differenced_dPTE,
        reverse_phase,
    )

    epochs = make_epochs()
    dPTE, raw = PTE_from_epochs(epochs, time_reversal=True)
    plain, plain_raw = PTE_from_epochs(epochs)

    np.testing.assert_array_equal(raw, plain_raw)
    data = epochs.get_data()
    stacked, delay, n_bins = prepare_phase(data.reshape(-1, data.shape[2]))
    first = stacked[:3]
    forward, _ = compute_dPTE_rawPTE(first, delay, n_bins=n_bins)
    backward, _ = compute_dPTE_rawPTE(reverse_phase(first), delay, n_bins=n_bins)
    np.testing.assert_allclose(dPTE[0], reversal_differenced_dPTE(forward, backward))

    result = group_test(dPTE)
    assert result.significant[0, 1] and result.effect[0, 1] > 0.5
