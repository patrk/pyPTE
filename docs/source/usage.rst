=====
Usage
=====

The main functionality of this package, calculating the Phase Transfer Entropy (PTE) for a set of time-series is accessible via the following functions:

If you are using MNE_ for analyzing EEG or MEG recordings an mne.io.Raw_ object can be passed to:

.. _MNE: https://www.martinos.org/mne/stable/index.html
.. _mne.io.Raw: https://martinos.org/mne/dev/generated/mne.io.Raw.html

.. code-block:: python

	from pyPTE.utils.mne_tools import PTE_from_mne
	
	dPTE, rPTE = PTE_from_mne(raw)

which returns a tuple of the normalized dPTE, containing information about the directionality and the raw PTE matrix, whereas the matrices are pandas DataFrames indexed by the channel names from the mne.io.Raw_ object.

In other domains the PTE calculation can be called directly by either passing a pandas.DataFrame_ to:

.. _pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

.. code-block:: python

	from pyPTE import pyPTE
	dPTE, rPTE = pyPTE.PTE_from_dataframe(dataframe)

or by passing a (m x n) numpy.ndarray_, where m is the number of samples and n is the number of time-series:

.. _numpy.ndarray: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.html
.. code-block:: python

	from pyPTE import pyPTE
	dPTE, rPTE = pyPTE.PTE(timeseries)

where the returned tuple consists of the above mentioned dPTE, rPTE matrices as (n x n) numpy.ndarray objects ordered in the same way as the input object.

If you are interested in further aspects of the implementation see Developer's documentation.
