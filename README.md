# pyPTE: Phase Transfer Entropy in Python

[![PyPI](https://img.shields.io/pypi/v/pyPTE.svg)](https://pypi.org/project/pyPTE/)
[![Python](https://img.shields.io/pypi/pyversions/pyPTE.svg)](https://pypi.org/project/pyPTE/)
[![CI](https://github.com/patrk/pyPTE/actions/workflows/main.yml/badge.svg)](https://github.com/patrk/pyPTE/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

`pyPTE` estimates **directed connectivity** between oscillatory signals: given
two channels of EEG, MEG or any rhythmic time series, it answers *which one is
leading the other*.

It is worth being precise about the claim, because it is narrower than it first
appears:

> **pyPTE recovers the direction of information flow between a pair of signals.
> It does not recover the wiring diagram of a network.**

Direction is reliable. Which connections are *direct* is not recoverable from a
bivariate measure, and [what can fool it](https://patrk.github.io/pyPTE/guide/pitfalls.html)
shows exactly how that fails, with numbers. Every claim below is produced by a
script in [`examples/`](examples/) that asserts its own result, so if something
here stops being true, the test suite says so.

Based on:

- Lobier et al., 2014: [Phase transfer entropy: A novel phase-based measure for directed connectivity in networks coupled by oscillatory interactions](http://dx.doi.org/10.1016/j.neuroimage.2013.08.056)
- Hillebrand et al., 2016: [Direction of information flow in large-scale resting-state networks is frequency-dependent](http://dx.doi.org/10.1073/pnas.1515657113)

---

## Installation

```bash
pip install pyPTE          # or: uv add pyPTE
```

The core needs only NumPy and SciPy. Adapters are optional:

```bash
pip install "pyPTE[mne]"      # MNE-Python adapter (also installs pandas)
pip install "pyPTE[pandas]"   # pandas adapter
```

Requires Python 3.11 or newer. Dependencies are deliberately unpinned at the top
end so `pyPTE` resolves alongside whatever NumPy and SciPy the rest of your
environment needs.

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

print(dPTE[0, 1])  # ~0.80 -> driver leads target
print(dPTE[1, 0])  # ~0.20 -> the reverse is suppressed
```

`raw_PTE` is transfer entropy in bits. `dPTE` is the direction-normalised form,
where `dPTE[i, j] + dPTE[j, i] == 1`: above `0.5` means net flow from `i` to
`j`, and exactly `0.5` means no preferred direction.

**A raw `dPTE` value on its own is not a finding.** See
[running a real analysis](https://patrk.github.io/pyPTE/guide/analysis.html).

## Documentation

**[patrk.github.io/pyPTE](https://patrk.github.io/pyPTE/)**

- [Tutorial](https://patrk.github.io/pyPTE/tutorial/how_pte_works.html): the measure built up from phase histograms, figure by figure
- [Running a real analysis](https://patrk.github.io/pyPTE/guide/analysis.html): surrogate tests, epoched group tests, multiple comparisons
- [What can fool it](https://patrk.github.io/pyPTE/guide/pitfalls.html): unequal noise, indirect paths, symmetric coupling
- [Examples](https://patrk.github.io/pyPTE/examples.html) and [API reference](https://patrk.github.io/pyPTE/api.html)

## Contributing

Contributions are welcome — issues, suggestions and pull requests alike.

```bash
git clone https://github.com/patrk/pyPTE && cd pyPTE
uv sync --all-extras
uv run pytest
```

Before opening a PR: `uv run ruff check .`, `uv run ruff format .`,
`uv run mypy pyPTE tests`.

## Citing

If pyPTE helps your research, please cite it. GitHub's "Cite this repository"
button gives the details from [CITATION.cff](CITATION.cff). Please also cite the
papers that introduced the method:

- Lobier, Siebenhühner, Palva & Palva (2014). Phase transfer entropy: a novel
  phase-based measure for directed connectivity in networks coupled by
  oscillatory interactions. *NeuroImage* 85, 853–872.
  [doi:10.1016/j.neuroimage.2013.08.056](https://doi.org/10.1016/j.neuroimage.2013.08.056)
- Hillebrand et al. (2016). Direction of information flow in large-scale
  resting-state networks is frequency-dependent. *PNAS* 113(14), 3867–3872.
  [doi:10.1073/pnas.1515657113](https://doi.org/10.1073/pnas.1515657113)

## License

`pyPTE` is released under GPL-3.0-or-later. See [LICENSE](LICENSE).
