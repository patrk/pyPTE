import mne


def interpolate_mne(raw, raw_reference):
    ref_channels = raw_reference.ch_chnames
    raw_channels = list(raw.ch_names)

    diff = list(channel for channel in ref_channels if channel not in raw_channels)

    b = raw.copy()
    d = raw.copy()

    if len(diff) > 0:
        b.pick_channels(b.ch_names[:len(diff)])
        ac = raw.copy().pick_channels(diff)
        b.info = ac.info
        d.add_channels([b])
        d.info['bads'] = diff
        d.interpolate_bads(verbose=False)

    return d