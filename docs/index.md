# pyPTE

Phase transfer entropy for directed connectivity between oscillatory signals.
Given two channels of EEG, MEG or any rhythmic time series, pyPTE answers
*which one is leading the other*.

> **pyPTE recovers the direction of information flow between a pair of signals.
> It does not recover the wiring diagram of a network.**

New here? Start with the [tutorial](tutorial/how_pte_works), which builds the
measure from phase histograms up, with every figure computed from a real run.

## Installation

```bash
pip install pyPTE          # or: uv add pyPTE
pip install "pyPTE[mne]"   # MNE-Python adapter (also installs pandas)
```

The core needs only NumPy and SciPy, and Python 3.11 or newer.

## Quickstart

`PTE` takes an `(n_channels, n_samples)` array and returns two matrices, where
entry `[i, j]` describes flow from channel `i` to channel `j`.

```python
import numpy as np
from pyPTE import PTE

rng = np.random.default_rng(0)
t = np.arange(8000) / 250.0
driver = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.standard_normal(t.size)
target = 0.9 * np.roll(driver, 12) + 0.4 * rng.standard_normal(t.size)

dPTE, raw_PTE = PTE(np.vstack([driver, target]))
dPTE[0, 1]  # above 0.5: driver leads target
```

`dPTE[i, j] + dPTE[j, i] == 1`: above 0.5 means net flow from `i` to `j`, and
0.5 means no preferred direction. **A dPTE value on its own is not a finding**;
test it as described in [running a real analysis](guide/analysis.md).

Based on Lobier et al. 2014, [Phase transfer entropy](http://dx.doi.org/10.1016/j.neuroimage.2013.08.056),
and Hillebrand et al. 2016, [Direction of information flow in large-scale resting-state networks](http://dx.doi.org/10.1073/pnas.1515657113).

```{toctree}
:hidden:

tutorial/index
guide/how-it-works
guide/analysis
guide/pitfalls
examples
api
contributing
```
