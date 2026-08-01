from typing import Any

import numpy as np
import pandas as pd

from pyPTE.core import pyPTE


def interpolate_mne(raw: Any, raw_reference: Any) -> Any:
    """
    Restores channels missing from a recording by interpolating them

    MNE can only interpolate channels that are present and marked as bad, so a
    recording that is missing channels outright cannot be repaired directly.
    This function inserts the missing channels as silent placeholders carrying
    the sensor positions of a reference recording, marks them bad, and lets
    mne.io.Raw.interpolate_bads() fill them in.

    Parameters
    ----------
    raw : mne.io.Raw
        The object missing channels to be interpolated
    raw_reference : mne.io.Raw
        The reference object containing all desired channels, used as the source
        of the sensor positions for the missing channels

    Returns
    -------
    interpolated : mne.io.Raw
        New object containing all information from the original raw object and
        interpolated channels

    Raises
    ------
    ValueError
        If the two recordings do not share a sampling frequency, in which case
        the placeholder channels could not be concatenated onto the timeline

    """
    import mne

    missing = [ch for ch in raw_reference.ch_names if ch not in raw.ch_names]
    if not missing:
        return raw.copy()

    if raw.info["sfreq"] != raw_reference.info["sfreq"]:
        raise ValueError(
            "raw and raw_reference must share a sampling frequency, got "
            f"{raw.info['sfreq']} and {raw_reference.info['sfreq']}"
        )

    # the placeholder samples are never read: interpolate_bads() overwrites every
    # channel listed in info["bads"] from the surrounding sensors
    placeholder_info = raw_reference.copy().pick(missing).info
    placeholders = mne.io.RawArray(
        np.zeros((len(missing), raw.n_times)),
        placeholder_info,
        first_samp=raw.first_samp,
        verbose=False,
    )

    interpolated = raw.copy().add_channels([placeholders], force_update_info=True)
    interpolated.info["bads"] = missing
    interpolated.interpolate_bads(verbose=False)
    return interpolated


def PTE_from_mne(raw: Any, picks: Any = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes dPTE and raw PTE matrices for the channels of an mne.io.Raw object

    Parameters
    ----------
    raw : mne.io.Raw
        EEG or MEG recording serving data in sensor space
    picks : str | list | slice | None
        Channels to analyse, accepting anything mne.io.Raw.pick() accepts. By
        default every channel is used, which for an unfiltered recording will
        include non-data channels such as stimulus triggers.

    Returns
    -------
    (dPTE_df, raw_PTE_df) : tuple of pandas.DataFrame objects
        The dPTE and raw PTE matrices, indexed in both dimensions by channel
        name, so entry ``[i, j]`` describes information flow from channel ``i``
        to channel ``j``

    """
    selection = raw.copy().pick(picks) if picks is not None else raw

    # mne serves data as (n_channels, n_times), which is what the core expects
    dPTE, raw_PTE = pyPTE.PTE(selection.get_data())

    channels = selection.ch_names
    dPTE_df = pd.DataFrame(dPTE, index=channels, columns=channels)
    raw_PTE_df = pd.DataFrame(raw_PTE, index=channels, columns=channels)
    return dPTE_df, raw_PTE_df
