def filter_by_subject(subject, data):
    return {k : v for k, v in data.items() if k.startswith(subject)}

def filter_by_session(session, data):
    return {k : v for k, v in data.items() if k.endswith(session)}

def get_common_subset(channel_list):
    common_subset = set.intersection(*map(set, channel_list))
    return common_subset

def get_missing_channels_mask(channels, common_subset):
    #mask = [1 if x in common_subset else 0 for x in channels]
    mask = [i for i, x in enumerate(channels) if x not in common_subset]

    for channel in channels:
        if channel not in common_subset:
            print('ommitted channel', channel)
    return mask

def get_channels_dict(measurements):
    channels_dict = dict()
    for key, value in measurements.items():
        print(key, value)
        channels_dict[key] = value.ch_names
        print(value.ch_names)
    return channels_dict

def get_all_masks(keys):
    channel_list = []
    for key in keys:
        # print(key)
        channels = raw[key].ch_names
        channel_list.append(channels)
    common_subset = get_common_subset(channel_list)

    masks = dict()
    for key in keys:
        channels = raw[key].ch_names
        masks[key] = get_missing_channels_mask(channels, common_subset)
    return masks