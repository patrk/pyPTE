from pandas_adapter import PTE_from_dataframe


def interpolate_mne(raw, raw_reference):
    """
    This is a utility function which circumvents the issue that MNE allows to
    interpolate only channels, which are present and marked as bad channels.
    Missing channels in a raw file can be interpolated by passing the mne.io.Raw object
    subject to interpolation and a reference object which contains all channels.
    This is achieved by copying channels, replacing its channel information by
    the reference channels, marking it as bad and finally utilizing
    mne.io.Raw.interpolate_bads()

    Parameters
    ----------
    raw : mne.io.Raw
        The object missing channels to be interpolated
    raw_reference : mne.io.Raw
        The reference object containing all desired channels

    Returns
    -------
    d : mne.io.Raw
        New object containing all information from the original raw object and
        interpolated channels

    """
    ref_channels = raw_reference.ch_chnames
    raw_channels = raw.ch_names

    diff = list(channel for channel in ref_channels if channel not in raw_channels)

    b = raw.copy()
    d = raw.copy()

    if len(diff) > 0:
        b.pick_channels(b.ch_names[: len(diff)])
        ac = raw.copy().pick_channels(diff)
        b.info = ac.info
        d.add_channels([b])
        d.info["bads"] = diff
        d.interpolate_bads(verbose=False)

    return d


def PTE_from_mne(mne_raw):
    """
    This is a wrapper which allows calculating dPTE,PTE matrices by passing an
    mne.io.Raw object and calling pyPTE.pyPTE.PTE_from_dataframe().

    Parameters
    ----------
    mne_raw : mne.io.Raw
        EEG or MEG recording serving data in sensor space

    Returns
    -------
    result : pandas.DataFrame
        The wrapper returns a tuple of (dPTE, PTE) matrices, which are stored as a
        pandas.DataFrame and indexed by the channels names of the input file.
        This allows convenient analysis of the results.

    """

    data_frame = mne_raw.to_data_frame()
    return PTE_from_dataframe(data_frame)
