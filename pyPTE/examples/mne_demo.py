import os.path as op

import mne
# from mne.datasets.brainstorm import bst_raw

data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'),
                          preload=True)
# raw.plot()

from pyPTE.utils import mne_tools

dPTE, rPTE = mne_tools.PTE_from_mne(raw)

import seaborn as sns
from matplotlib import pyplot as plt

sns.heatmap(rPTE)


