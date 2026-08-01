# pyPTE: Phase Transfer Entropy in Python

`pyPTE` is an open-source Python implementation of the Phase Transfer Entropy method, designed to analyze directed connectivity in networks influenced by oscillatory interactions. This tool is inspired by the following foundational works:

- Lobier et al., 2014: [Phase transfer entropy: A novel phase-based measure for directed connectivity in networks coupled by oscillatory interactions](http://dx.doi.org/10.1016/j.neuroimage.2013.08.056).
- Hillebrand et al., 2016: [Direction of information flow in large-scale resting-state networks is frequency-dependent](http://dx.doi.org/10.1073/pnas.1515657113).

## Installation

```bash
pip install pyPTE          # or: uv add pyPTE
```

The core requires only NumPy and SciPy. The adapters are optional:

```bash
pip install "pyPTE[mne]"      # MNE-Python adapter (also installs pandas)
pip install "pyPTE[pandas]"   # pandas adapter
```

Requires Python 3.11 or newer.

## Quickstart

`PTE` takes an `(n_channels, n_samples)` array and returns two matrices, where
entry `[i, j]` describes information flow from channel `i` to channel `j`.

```python
import numpy as np
from pyPTE.core.pyPTE import PTE

rng = np.random.default_rng(0)
t = np.arange(8000) / 250.0

driver = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.standard_normal(t.size)
target = 0.9 * np.roll(driver, 12) + 0.4 * rng.standard_normal(t.size)

dPTE, raw_PTE = PTE(np.vstack([driver, target]))

print(dPTE[0, 1])  # ~0.80 -> driver leads target
print(dPTE[1, 0])  # ~0.20 -> the reverse direction is suppressed
```

`raw_PTE` holds the transfer entropy in bits. `dPTE` is the direction-normalised
form, where `dPTE[i, j] + dPTE[j, i] == 1`: values above `0.5` mean net flow
from `i` to `j`, and `0.5` means no preferred direction.

## Introduction

Phase Transfer Entropy (PTE) is a measure for directed connectivity in networks coupled by oscillatory interactions. The `pyPTE` library provides a Python implementation of this method, allowing researchers and developers to apply PTE analysis to their data.

### Mathematical Background

The mathematical formulation of PTE can be described as follows:

Given two time series $`X`$ and $`Y`$, the PTE is defined as:

$$PTE_{X \to Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)$$

The PTE value quantifies the amount of uncertainty reduced in predicting the future phase of $`Y`$ when considering the current phase of $`X`$.

## Contributing
Contributions to `pyPTE` are welcome! If you have suggestions, bug reports, or want to contribute code via Pull Requests

## License

`pyPTE` is released under the GPL-3.0 license. For more details, see the [LICENSE](https://github.com/patrk/pyPTE/blob/master/LICENSE) file.





