import os.path as op

import mne
# from mne.datasets.brainstorm import bst_raw

data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'),
                          preload=True)
# raw.plot()

from pyPTE.adapters import mne_adapter

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
t_idx = raw.time_as_index([10., 20.])
data, times = raw[picks, t_idx[0]:t_idx[1]]


dPTE, rPTE = mne_adapter.PTE_from_mne(raw)

import seaborn as sns
from matplotlib import pyplot as plt

sns.heatmap(rPTE)




